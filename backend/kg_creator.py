from neo4j import GraphDatabase
import os
import uuid
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Any, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('kg_creator')

load_dotenv()

try:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info("Successfully connected to Neo4j")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    driver = None

try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    logger.info("Successfully initialized LLM")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    llm = None

STANCE_MAP = {
    "FOR": "SUPPORTS", 
    "AGAINST": "OPPOSES", 
    "NEUTRAL": "NEUTRAL"
}

argument_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Extract the main arguments from this Reddit content.
Return them as a numbered list with ONLY the arguments, no explanatory text.
If there are no clear arguments, return "No clear arguments found".

Content: {text}"""
)

argument_chain = LLMChain(llm=llm, prompt=argument_extraction_prompt) if llm else None

class KGRequest(BaseModel):
    thread_data: Dict[str, Any]

def create_knowledge_graph(thread_data: dict) -> dict:
    if not driver:
        return {"status": "error", "message": "Neo4j driver not initialized"}

    discussion_id = str(uuid.uuid4())
    comment_count = 0
    reply_count = 0
    argument_count = 0

    try:
        topic_title = thread_data['post'].get('title', 'Unknown Topic')
        logger.info(f"Creating knowledge graph for topic: {topic_title} | discussion_id: {discussion_id}")

        if "classified_comments" not in thread_data:
            return {"status": "error", "message": "No classified_comments found in thread_data"}

        with driver.session() as session:
            session.execute_write(merge_topic, topic_title, discussion_id)
            logger.info(f"Created topic node: {topic_title}")

            for stance in ["FOR", "AGAINST", "NEUTRAL"]:
                for i, comment in enumerate(thread_data["classified_comments"].get(stance, [])):
                    if "id" not in comment or not comment["id"]:
                        comment["id"] = f"comment_{stance}_{i}"
                    comment.update({"discussion_id": discussion_id})
                    session.execute_write(merge_comment, comment)
                    comment_count += 1
                    session.execute_write(connect_comment_to_topic, comment["id"], topic_title, stance, discussion_id)

                    for j, reply in enumerate(comment.get("replies", [])):
                        if "id" not in reply or not reply["id"]:
                            reply["id"] = f"reply_{comment['id']}_{j}"
                        else:
                            reply["id"] = f"{comment['id']}_{reply['id']}"
                        reply.update({"discussion_id": discussion_id})
                        session.execute_write(merge_reply, reply)
                        reply_count += 1
                        session.execute_write(connect_reply_to_comment, reply["id"], comment["id"], discussion_id)

                        if argument_chain:
                            arguments = extract_arguments(reply.get("body", ""))
                            for arg in arguments:
                                arg = arg.strip()
                                session.execute_write(merge_argument, arg, reply.get("stance", "NEUTRAL"), discussion_id)
                                session.execute_write(connect_argument_to_reply, arg, reply["id"], discussion_id)
                                argument_count += 1

                    if argument_chain:
                        arguments = extract_arguments(comment.get("body", ""))
                        for arg in arguments:
                            arg = arg.strip()
                            session.execute_write(merge_argument, arg, stance, discussion_id)
                            session.execute_write(connect_argument_to_comment, arg, comment["id"], discussion_id)
                            argument_count += 1

        logger.info(f"Graph created: {comment_count} comments, {reply_count} replies, {argument_count} arguments.")
        return {
            "status": "success",
            "discussion_id": discussion_id,
            "nodes_created": {
                "topic": 1,
                "comments": comment_count,
                "replies": reply_count,
                "arguments": argument_count
            }
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"status": "error", "message": str(e)}

def extract_arguments(text: str) -> List[str]:
    try:
        if not argument_chain:
            return []
        response = argument_chain.run(text=text)
        if "No clear arguments found" in response:
            return []
        return [line.split(" ", 1)[-1].strip() for line in response.strip().split('\n') if line.strip()]
    except Exception as e:
        logger.error(f"Argument extraction error: {e}")
        return []

def merge_topic(tx, title, discussion_id):
    tx.run("""
        MERGE (t:Topic {title: $title, discussion_id: $discussion_id})
        SET t.created_at = datetime(), t.updated_at = datetime()
    """, title=title, discussion_id=discussion_id)

def merge_comment(tx, comment):
    tx.run("""
        MERGE (c:Comment {id: $id, discussion_id: $discussion_id})
        SET c.body = $body, c.author = $author, c.score = $score, c.updated_at = datetime()
    """, **comment)

def connect_comment_to_topic(tx, comment_id, topic_title, stance, discussion_id):
    relationship = STANCE_MAP[stance]
    tx.run(f"""
        MATCH (t:Topic {{title: $topic_title, discussion_id: $discussion_id}})
        MATCH (c:Comment {{id: $comment_id, discussion_id: $discussion_id}})
        MERGE (c)-[:{relationship}]->(t)
    """, topic_title=topic_title, comment_id=comment_id, discussion_id=discussion_id)

def merge_reply(tx, reply):
    tx.run("""
        MERGE (r:Reply {id: $id, discussion_id: $discussion_id})
        SET r.body = $body, r.author = $author, r.score = $score, r.updated_at = datetime()
    """, **reply)

def connect_reply_to_comment(tx, reply_id, comment_id, discussion_id):
    tx.run("""
        MATCH (r:Reply {id: $reply_id, discussion_id: $discussion_id})
        MATCH (c:Comment {id: $comment_id, discussion_id: $discussion_id})
        MERGE (r)-[:REPLY_TO]->(c)
    """, reply_id=reply_id, comment_id=comment_id, discussion_id=discussion_id)

def merge_argument(tx, text, stance, discussion_id):
    tx.run("""
        MERGE (a:Argument {text: $text, discussion_id: $discussion_id})
        SET a.stance = $stance, a.updated_at = datetime()
    """, text=text, stance=stance, discussion_id=discussion_id)

def connect_argument_to_comment(tx, text, comment_id, discussion_id):
    tx.run("""
        MATCH (a:Argument {text: $text, discussion_id: $discussion_id})
        MATCH (c:Comment {id: $comment_id, discussion_id: $discussion_id})
        MERGE (a)-[:EXTRACTED_FROM]->(c)
    """, text=text, comment_id=comment_id, discussion_id=discussion_id)

def connect_argument_to_reply(tx, text, reply_id, discussion_id):
    tx.run("""
        MATCH (a:Argument {text: $text, discussion_id: $discussion_id})
        MATCH (r:Reply {id: $reply_id, discussion_id: $discussion_id})
        MERGE (a)-[:EXTRACTED_FROM]->(r)
    """, text=text, reply_id=reply_id, discussion_id=discussion_id)
