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

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('kg_creator')

load_dotenv()

driver = None
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

llm = None
try:
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    logger.info("Successfully initialized LLM")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")

STANCE_MAP = {
    "FOR": "SUPPORTS",
    "AGAINST": "OPPOSES",
    "NEUTRAL": "NEUTRAL"
}

argument_extraction_prompt = PromptTemplate(
    input_variables=["text", "topic"],
    template="""
    You are an AI trained to extract arguments from Reddit discussions.

    Given a Reddit comment or reply and the main discussion topic, extract ONLY the arguments clearly expressed in the text.
    Return them as a numbered list with ONLY the arguments, no explanatory text.

    CRITICAL INSTRUCTIONS:
    - Each argument must be SELF-CONTAINED: It should make complete sense on its own, without requiring the reader to see the original comment.
    - Do NOT include vague phrases like "this" or "it" without clarifying what is being referred to.
    - Avoid passive constructions unless they are fully understandable.
    - Exclude irrelevant content or background information. Focus only on specific opinions, reasons, or justifications that support or oppose the topic.
    - Express arguments clearly and concisely.
    - If there are no clear arguments, return: "No clear arguments found."

    Reddit Topic: "{topic}"

    Content: {text}
    """
)

argument_chain = LLMChain(llm=llm, prompt=argument_extraction_prompt) if llm else None

class KGRequest(BaseModel):
    thread_data: Dict[str, Any]

def create_knowledge_graph(thread_data: dict) -> dict:
    if not driver:
        return {"status": "error", "message": "Neo4j driver not initialized"}

    comment_count = 0
    reply_count = 0
    argument_count = 0

    try:
        topic_title = thread_data['post'].get('title', 'Unknown Topic')
        thread_url = thread_data['post'].get('url', 'unknown')

        with driver.session() as session:
            existing = session.execute_read(check_existing_discussion_by_url, thread_url)
            discussion_id = existing if existing else str(uuid.uuid4())
            if existing:
                logger.info(f"Graph already exists for URL: {thread_url} (discussion_id: {existing}), updating...")
            else:
                logger.info(f"Creating new graph for: {topic_title} | discussion_id: {discussion_id}")

            session.execute_write(merge_topic, topic_title, discussion_id, thread_url)

            for stance in ["FOR", "AGAINST", "NEUTRAL"]:
                for i, comment in enumerate(thread_data["classified_comments"].get(stance, [])):
                    if "id" not in comment or not comment["id"]:
                        comment["id"] = f"comment_{stance}_{i}"
                    comment.update({"discussion_id": discussion_id})

                    # Check if comment exists in the database
                    comment_exists = session.execute_read(check_comment_exists, comment["id"], discussion_id)
                    session.execute_write(merge_comment, comment)
                    
                    # Only create new relationships and extract arguments if this is a new comment
                    if not comment_exists:
                        comment_count += 1
                        session.execute_write(connect_comment_to_topic, comment["id"], topic_title, stance, discussion_id)

                        if argument_chain:
                            arguments = extract_arguments(comment.get("body", ""), topic_title)
                            for arg in arguments:
                                arg = arg.strip()
                                session.execute_write(merge_argument, arg, stance, discussion_id)
                                session.execute_write(connect_argument_to_comment, arg, comment["id"], discussion_id)
                                argument_count += 1

                    for j, reply in enumerate(comment.get("replies", [])):
                        if "id" not in reply or not reply["id"]:
                            reply["id"] = f"reply_{comment['id']}_{j}"
                        else:
                            reply["id"] = f"{comment['id']}_{reply['id']}"
                        reply.update({"discussion_id": discussion_id})

                        # Check if reply exists in the database
                        reply_exists = session.execute_read(check_reply_exists, reply["id"], discussion_id)
                        session.execute_write(merge_reply, reply)
                        
                        # Only create new relationships and extract arguments if this is a new reply
                        if not reply_exists:
                            reply_count += 1
                            session.execute_write(connect_reply_to_comment, reply["id"], comment["id"], discussion_id)

                            if argument_chain:
                                arguments = extract_arguments(reply.get("body", ""), topic_title)
                                for arg in arguments:
                                    arg = arg.strip()
                                    session.execute_write(merge_argument, arg, reply.get("stance", "NEUTRAL"), discussion_id)
                                    session.execute_write(connect_argument_to_reply, arg, reply["id"], discussion_id)
                                    argument_count += 1

        logger.info(f"Graph update complete: {comment_count} new comments, {reply_count} new replies, {argument_count} new arguments.")
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

def extract_arguments(text: str, topic: str) -> List[str]:
    try:
        if not argument_chain:
            return []
        response = argument_chain.run(text=text, topic=topic)
        if "No clear arguments found" in response:
            return []
        return [line.split(" ", 1)[-1].strip() for line in response.strip().split('\n') if line.strip()]
    except Exception as e:
        logger.error(f"Argument extraction error: {e}")
        return []

def check_existing_discussion_by_url(tx, url: str):
    result = tx.run("""
        MATCH (t:Topic {url: $url})
        RETURN t.discussion_id AS discussion_id
    """, url=url)
    record = result.single()
    return record["discussion_id"] if record else None

def check_comment_exists(tx, comment_id, discussion_id):
    result = tx.run("""
        MATCH (c:Comment {id: $comment_id, discussion_id: $discussion_id})
        RETURN count(c) > 0 AS exists
    """, comment_id=comment_id, discussion_id=discussion_id)
    return result.single()["exists"]

def check_reply_exists(tx, reply_id, discussion_id):
    result = tx.run("""
        MATCH (r:Reply {id: $reply_id, discussion_id: $discussion_id})
        RETURN count(r) > 0 AS exists
    """, reply_id=reply_id, discussion_id=discussion_id)
    return result.single()["exists"]

def merge_topic(tx, title, discussion_id, url):
    tx.run("""
        MERGE (t:Topic {url: $url})
        SET t.title = $title,
            t.discussion_id = $discussion_id,
            t.updated_at = datetime()
    """, title=title, discussion_id=discussion_id, url=url)

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