from neo4j import GraphDatabase
import os
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

# Configuration: Ensure that we are connecting to the correct Neo4j instance.
try:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    # Test connection
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

# Constants for relationship types
STANCE_MAP = {
    "FOR": "SUPPORTS", 
    "AGAINST": "OPPOSES", 
    "NEUTRAL": "NEUTRAL"
}

# LLM Prompt for Argument Extraction - simplified
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
    """
    Create a knowledge graph with:
      - Topic node (center).
      - Comment nodes connected to Topic using stance-based relationships.
      - Reply nodes connected to their parent Comment (REPLY_TO).
      - Argument nodes connected to either Comment or Reply nodes (EXTRACTED_FROM).
    """
    if not driver:
        return {"status": "error", "message": "Neo4j driver not initialized"}
    
    try:
        topic_title = thread_data['post'].get('title', 'Unknown Topic')
        logger.info(f"Creating knowledge graph for topic: {topic_title}")
        
        if "classified_comments" not in thread_data:
            logger.error("No classified_comments found in thread_data")
            return {"status": "error", "message": "No classified_comments found in thread_data"}
        
        for stance in ["FOR", "AGAINST", "NEUTRAL"]:
            count = len(thread_data["classified_comments"].get(stance, []))
            logger.info(f"Found {count} comments with stance {stance}")
        
        comment_count = 0
        reply_count = 0
        argument_count = 0
        
        with driver.session() as session:
            # Create Topic Node
            topic_id = session.execute_write(merge_topic, topic_title)
            logger.info(f"Created topic node with ID: {topic_id}")

            for stance in ["FOR", "AGAINST", "NEUTRAL"]:
                for i, comment in enumerate(thread_data["classified_comments"].get(stance, [])):
                    try:
                        if "id" not in comment:
                            logger.warning(f"Comment missing ID field: {comment}")
                            comment["id"] = f"comment_{stance}_{i}"
                        # Set default values for comment fields if missing.
                        comment["body"] = comment.get("body", "")
                        comment["author"] = comment.get("author", "unknown")
                        comment["score"] = comment.get("score", 0)
                        
                        # Create Comment Node.
                        comment_id = session.execute_write(merge_comment, comment)
                        comment_count += 1
                        logger.info(f"Created comment node with ID: {comment_id}")
                        
                        # Connect comment to topic.
                        session.execute_write(connect_comment_to_topic, comment["id"], topic_title, stance)
                        logger.info(f"Connected comment {comment['id']} to topic with relationship: {STANCE_MAP[stance]}")
                        
                        # Process replies.
                        replies = comment.get("replies", [])
                        # If replies is not a list, check if it's a dict with a 'children' key.
                        if not isinstance(replies, list):
                            if isinstance(replies, dict):
                                replies = replies.get("children", [])
                            else:
                                replies = []
                        logger.info(f"Comment {comment['id']} has {len(replies)} replies")
                        for j, reply in enumerate(replies):
                            try:
                                if "id" not in reply:
                                    reply["id"] = f"reply_{comment['id']}_{j}"
                                reply["body"] = reply.get("body", "")
                                reply["author"] = reply.get("author", "unknown")
                                reply["score"] = reply.get("score", 0)
                                reply_stance = reply.get("stance", "NEUTRAL")
                                
                                # Create Reply Node.
                                reply_id = session.execute_write(merge_reply, reply)
                                reply_count += 1
                                logger.info(f"Created reply node with ID: {reply_id} for comment {comment['id']}")
                                
                                # Connect reply to its parent comment.
                                session.execute_write(connect_reply_to_comment, reply["id"], comment["id"])
                                logger.info(f"Connected reply {reply['id']} to comment {comment['id']}")
                                
                                # Process arguments from reply.
                                if argument_chain:
                                    arguments = extract_arguments(reply["body"])
                                    if arguments:
                                        logger.info(f"Extracted {len(arguments)} arguments from reply {reply['id']}")
                                        for arg in arguments:
                                            arg_clean = arg.strip()
                                            session.execute_write(merge_argument, arg_clean, reply_stance)
                                            argument_count += 1
                                            session.execute_write(connect_argument_to_reply, arg_clean, reply["id"])
                                            logger.info(f"Connected argument '{arg_clean}' to reply {reply['id']}")
                            except Exception as e:
                                logger.error(f"Error processing reply for comment {comment['id']}: {e}")
                        
                        # Process arguments for the comment.
                        if argument_chain:
                            arguments = extract_arguments(comment["body"])
                            if arguments:
                                logger.info(f"Extracted {len(arguments)} arguments from comment {comment['id']}")
                                for arg in arguments:
                                    arg_clean = arg.strip()
                                    session.execute_write(merge_argument, arg_clean, stance)
                                    argument_count += 1
                                    session.execute_write(connect_argument_to_comment, arg_clean, comment["id"])
                                    logger.info(f"Connected argument '{arg_clean}' to comment {comment['id']}")
                    except Exception as e:
                        logger.error(f"Error processing comment: {e}")
            
            logger.info(f"Knowledge graph creation complete: {comment_count} comments, {reply_count} replies, {argument_count} arguments created.")
            return {
                "status": "success",
                "nodes_created": {
                    "topic": 1,
                    "comments": comment_count,
                    "replies": reply_count,
                    "arguments": argument_count
                }
            }
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {e}")
        return {"status": "error", "message": str(e)}

def extract_arguments(text: str) -> List[str]:
    """Extract discrete arguments from text using the LLM."""
    try:
        if not argument_chain:
            return []
        response = argument_chain.run(text=text)
        if "No clear arguments found" in response:
            return []
        args = []
        for line in response.strip().split('\n'):
            clean_line = line.strip()
            if clean_line:
                if clean_line[0].isdigit():
                    parts = clean_line.split(' ', 1)
                    if len(parts) > 1 and parts[0].rstrip('.):').isdigit():
                        clean_line = parts[1].strip()
                if clean_line.startswith('- ') or clean_line.startswith('• '):
                    clean_line = clean_line[2:].strip()
                if clean_line:
                    args.append(clean_line)
        return args
    except Exception as e:
        logger.error(f"Error extracting arguments: {e}")
        return []

def merge_topic(tx, title: str):
    """Create or update a Topic node and return its ID."""
    result = tx.run("""
        MERGE (t:Topic {title: $title})
        ON CREATE SET t.created_at = datetime(), t.updated_at = datetime()
        ON MATCH SET t.updated_at = datetime()
        RETURN t.title AS id
    """, title=title)
    return result.single()["id"]

def merge_comment(tx, comment: dict):
    """Create or update a Comment node and return its ID."""
    result = tx.run("""
        MERGE (c:Comment {id: $id})
        SET c.body = $body,
            c.author = $author,
            c.score = $score,
            c.updated_at = datetime()
        RETURN c.id AS id
    """, 
    id=comment["id"],
    body=comment["body"],
    author=comment["author"],
    score=comment["score"]
    )
    return result.single()["id"]

def connect_comment_to_topic(tx, comment_id: str, topic_title: str, stance: str):
    """Connect a Comment to a Topic with the appropriate stance relationship."""
    relationship_type = STANCE_MAP[stance]
    tx.run(f"""
        MATCH (t:Topic {{title: $topic_title}})
        MATCH (c:Comment {{id: $comment_id}})
        MERGE (c)-[:{relationship_type}]->(t)
    """, topic_title=topic_title, comment_id=comment_id)

def merge_reply(tx, reply: dict):
    """Create or update a Reply node and return its ID."""
    result = tx.run("""
        MERGE (r:Reply {id: $id})
        SET r.body = $body,
            r.author = $author,
            r.score = $score,
            r.updated_at = datetime()
        RETURN r.id AS id
    """, 
    id=reply["id"],
    body=reply["body"],
    author=reply["author"],
    score=reply["score"]
    )
    return result.single()["id"]

def connect_reply_to_comment(tx, reply_id: str, comment_id: str):
    """Connect a Reply to its parent Comment using REPLY_TO relationship."""
    tx.run("""
        MATCH (r:Reply {id: $reply_id})
        MATCH (c:Comment {id: $comment_id})
        MERGE (r)-[:REPLY_TO]->(c)
    """, reply_id=reply_id, comment_id=comment_id)

def merge_argument(tx, argument_text: str, stance: str):
    """Create or update an Argument node and return its ID."""
    result = tx.run("""
        MERGE (a:Argument {text: $text})
        SET a.stance = $stance,
            a.updated_at = datetime()
        RETURN a.text AS id
    """, text=argument_text, stance=stance)
    return result.single()["id"]

def connect_argument_to_comment(tx, argument_text: str, comment_id: str):
    """Connect an Argument to its source Comment using EXTRACTED_FROM relationship."""
    tx.run("""
        MATCH (a:Argument {text: $text})
        MATCH (c:Comment {id: $comment_id})
        MERGE (a)-[:EXTRACTED_FROM]->(c)
    """, text=argument_text, comment_id=comment_id)

def connect_argument_to_reply(tx, argument_text: str, reply_id: str):
    """Connect an Argument to its source Reply using EXTRACTED_FROM relationship."""
    tx.run("""
        MATCH (a:Argument {text: $text})
        MATCH (r:Reply {id: $reply_id})
        MERGE (a)-[:EXTRACTED_FROM]->(r)
    """, text=argument_text, reply_id=reply_id)
