from neo4j import GraphDatabase
import os
import uuid
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Any, Dict, List, Tuple
import logging


# Load environment variables
load_dotenv()

logger = logging.getLogger('kg_creator')

# Initialize Neo4j driver
driver = None
try:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    # Test the connection by running a dummy query
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info("Successfully connected to Neo4j")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")

# Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Stance mapping
STANCE_MAP = {
    "FOR": "SUPPORTS",
    "AGAINST": "OPPOSES",
    "NEUTRAL": "NEUTRAL"
}

# Prompt to extract arguments
argument_extraction_prompt = PromptTemplate(
    input_variables=["text", "topic"],
    template="""
    You are an AI trained to extract formal, self-contained, **non-redundant arguments** from Reddit discussions.

    Your task is to extract only arguments that can be used in scientific, academic, or logical contexts. 
    Arguments should clearly support or oppose a specific claim related to the discussion topic. 
    Avoid insults, and vague statements.

    ### INSTRUCTIONS:
    - Extract arguments that include **reasoning, causal relationships, or factual claims**.
    - Each argument must be **self-contained**: don't use pronouns like "this" or "it" without defining them.
    - **Avoid repeating the same idea** in different words.
    - Do **not** extract vague, general statements or insults.
    - If no clear arguments exist, return: **"No clear arguments found."**

    ### EXAMPLES:

    Topic: "Should schools ban smartphones?"
    Text:
    Parents that give kids smarthphones are stupid. Smartphones distract students from learning.  
    Kids use them during class to cheat on tests.  
    Smartphones allow students to stay connected with parents in emergencies.

    Output:
    1. Smartphones distract students from learning.
    2. Students use smartphones to cheat during exams.
    3. Smartphones can help students stay in touch with parents during emergencies.

    ---

    Topic: "Is Trump a threat to democracy?"
    Text:
    Trump is very stupid.  
    Trump is cunning and wants revenge.  
    He pressures officials to do what he wants.  
    Trump uses his power to discredit investigations.

    Output:
    1. Trump has pressured government officials to influence investigations.
    2. Trump has used his power to discredit investigations and investigators.
    3. Trump seeks to consolidate power for personal gain, undermining democratic norms.

    ---

    ### Now extract arguments for the following post:

    Reddit Topic: "{topic}"

    Content:
    {text}

    Only output a numbered list of arguments, or "No clear arguments found."
    """
)

# Build the argument extraction chain
argument_chain = argument_extraction_prompt | llm

# Prompt to classify arguments
stance_classification_prompt = PromptTemplate(
    input_variables=["argument", "topic"],
    template="""
    You are given the Reddit discussion topic:
    "{topic}"

    Classify the stance expressed in the following argument with one of these labels: FOR, AGAINST, or NEUTRAL.

    Argument:
    "{argument}"

    Instructions:
    - FOR: The argument expresses clear or implicit support for the topic.
    - AGAINST: The argument expresses clear or implicit opposition to the topic.
    - NEUTRAL: The argument neither supports nor opposes the topic, or is ambiguous.

    Consider the overall position conveyed, including implicit meanings or nuances, not only explicit statements.

    Return exactly one word: FOR, AGAINST, or NEUTRAL.
    """
)

stance_classifier = stance_classification_prompt | llm

#Prompt to create argument clusters
argument_grouping_prompt = PromptTemplate(
    input_variables=["stance", "arguments"],
    template="""
    You are an AI system trained to group semantically similar arguments from Reddit discussions.

    ### Objective:
    Given a list of arguments that all share the same stance ("{stance}") toward a topic, group those that express the same specific idea — even if they use different wording. 
    For each group, write a **detailed and self-contained summary** that explains the **shared claim** and the **main reasoning** behind it.

    ### Instructions:
    - The summary must make it **completely clear what the core argument is** without reading the individual comments.
    - Be specific. Clearly express what the grouped users believe or argue, and **why** they argue it.
    - Do NOT use vague generalizations like "Criticism of the left" or "Concerns about corruption."
    - Avoid summarizing with abstract categories — instead, explain the **actual position** being taken.
    - Only group arguments that express the same idea. Do not merge arguments that are different, even if related.
    - Use the original wording for the bullet points. Do not rewrite or paraphrase them.
    - If an argument doesn't belong in a group, place it alone with its own summary.

    ### Format:

    Group: <detailed explanation of the shared argument and reasoning>
    - <original argument 1>
    - <original argument 2>

    Group: <another detailed explanation>
    - <original argument 3>

    ### Arguments:
    {arguments}
    """
)

argument_grouping_chain = argument_grouping_prompt | llm

# Pydantic model
class KGRequest(BaseModel):
    thread_data: Dict[str, Any]

# Extrai argumentos e classifica a stance de cada um
def extract_and_classify_arguments(text: str, topic: str) -> List[Tuple[str, str]]:
    response = argument_chain.invoke({"text": text, "topic": topic})
    content = response.content.strip()

    if "No clear arguments found" in content:
        return []

    extracted = [line.split(" ", 1)[-1].strip() for line in content.split('\n') if line.strip()]
    result = []

    for arg in extracted:
        stance_resp = stance_classifier.invoke({"argument": arg, "topic": topic})
        stance = stance_resp.content.strip().upper()
        if stance in ["FOR", "AGAINST", "NEUTRAL"]:
            result.append((arg, stance))
        else:
            logger.warning(f"Invalid stance returned for argument: {arg} -> {stance}")

    return result

# Debug function to see what's in the database
def debug_existing_content(discussion_id):
    """Debug function to see what content exists in the database"""
    with driver.session() as session:
        # Check comments
        comment_result = session.run("""
            MATCH (c:Comment {discussion_id: $discussion_id})
            RETURN c.id AS comment_id
            ORDER BY c.id
        """, discussion_id=discussion_id)
        
        comments = [record["comment_id"] for record in comment_result]
        logger.info(f"Existing comments: {comments}")
        
        # Check replies
        reply_result = session.run("""
            MATCH (r:Reply {discussion_id: $discussion_id})
            RETURN r.id AS reply_id, r.parent_comment_id AS parent_id
            ORDER BY r.id
        """, discussion_id=discussion_id)
        
        replies = [(record["reply_id"], record["parent_id"]) for record in reply_result]
        logger.info(f"Existing replies: {replies}")

# Knowledge graph creation
def create_knowledge_graph(thread_data: dict) -> dict:
    # Get post title and URL (used as unique discussion ID)
    topic_title = thread_data['post'].get('title', 'Unknown Topic')
    thread_url = thread_data['post'].get('url', 'unknown')
    comment_count = 0
    reply_count = 0
    argument_count = 0
    new_content_added = False  # Track if any new content was added

    with driver.session() as session:
        # Check if the thread already exists in the graph
        existing = session.execute_read(check_existing_discussion_by_url, thread_url)
        # Ensure the topic node is present in the graph
        discussion_id = existing if existing else str(uuid.uuid4())
        session.execute_write(merge_topic, topic_title, discussion_id, thread_url)

        # Debug: Show what exists before processing
        logger.info(f"Processing discussion: {discussion_id}")
        debug_existing_content(discussion_id)

        # Loop through each stance category: FOR, AGAINST, NEUTRAL
        for stance in ["FOR", "AGAINST", "NEUTRAL"]:
            for i, comment in enumerate(thread_data["classified_comments"].get(stance, [])):
                # Ensure each comment has a unique ID - use original Reddit ID
                original_comment_id = comment.get("id", "")
                if not original_comment_id:
                    comment["id"] = f"comment_{stance}_{i}"
                else:
                    comment["id"] = original_comment_id
                    
                comment.update({"discussion_id": discussion_id})

                # Check if the comment is already in the database
                comment_exists = session.execute_read(check_comment_exists, comment["id"], discussion_id)
                logger.info(f"Comment {comment['id']} exists: {comment_exists}")
                
                session.execute_write(merge_comment, comment)
                
                # Only process new comments
                if not comment_exists:
                    logger.info(f"Adding NEW comment: {comment['id']}")
                    new_content_added = True
                    comment_count += 1
                    session.execute_write(connect_comment_to_topic, comment["id"], topic_title, stance, discussion_id)

                    # Extract arguments from the comment and store them in the graph
                    arguments_with_stance = extract_and_classify_arguments(comment.get("body", ""), topic_title)
                    for arg, stance_classified in arguments_with_stance:
                        session.execute_write(merge_argument, arg, stance_classified, discussion_id)
                        session.execute_write(connect_argument_to_comment, arg, comment["id"], discussion_id)
                        argument_count += 1
                
                # Handle replies for the comment
                for j, reply in enumerate(comment.get("replies", [])):
                    # CRITICAL: Use deterministic ID generation - same input = same output
                    original_reply_id = reply.get("id", "")
                    
                    if original_reply_id:
                        # Simple, consistent format that never changes
                        reply["id"] = f"reply_{original_reply_id}"
                    else:
                        # Deterministic fallback using comment ID and position
                        reply["id"] = f"reply_{comment['id']}_{j}"
                    
                    reply.update({
                        "discussion_id": discussion_id,
                        "parent_comment_id": comment["id"]
                    })

                    # Debug logging
                    logger.info(f"Processing reply ID: {reply['id']} for comment: {comment['id']}")
                    
                    # Check if the reply is already in the database
                    reply_exists = session.execute_read(check_reply_exists, reply["id"], discussion_id)
                    logger.info(f"Reply {reply['id']} already exists: {reply_exists}")
                    
                    # Always merge (update if exists, create if not)
                    session.execute_write(merge_reply, reply)

                    # Only process new replies
                    if not reply_exists:
                        logger.info(f"Adding NEW reply: {reply['id']}")
                        new_content_added = True
                        reply_count += 1
                        session.execute_write(connect_reply_to_comment, reply["id"], comment["id"], discussion_id)

                        # Extract and add arguments from the reply
                        arguments_with_stance = extract_and_classify_arguments(reply.get("body", ""), topic_title)
                        for arg, stance_classified in arguments_with_stance:
                            session.execute_write(merge_argument, arg, stance_classified, discussion_id)
                            session.execute_write(connect_argument_to_reply, arg, reply["id"], discussion_id)
                            argument_count += 1
                    else:
                        logger.info(f"Skipping existing reply: {reply['id']}")

        # Only regroup arguments if new content was added
        if new_content_added:
            logger.info("New content was added - regrouping arguments")
            group_arguments_by_stance(discussion_id)
        else:
            logger.info("No new content added - skipping argument regrouping")
    
    return {
        "status": "success",
        "discussion_id": discussion_id,
        "nodes_created": {
            "topic": 1,
            "comments": comment_count,
            "replies": reply_count,
            "arguments": argument_count
        },
        "new_content_added": new_content_added
    }


def group_arguments_by_stance(discussion_id: str):
    if not argument_grouping_chain or not driver:
        return

    with driver.session() as session:
        # First, clear existing argument groups for this discussion
        session.run("""
            MATCH (a:Argument {discussion_id: $discussion_id})-[r:HAS_GROUP]->(g:ArgumentGroup)
            DELETE r
            WITH DISTINCT g
            WHERE NOT (g)<-[:HAS_GROUP]-()
            DELETE g
        """, discussion_id=discussion_id)

        # Loop through each stance
        for stance in ["FOR", "AGAINST", "NEUTRAL"]:
            # Get all arguments for this stance and discussion
            result = session.run("""
                MATCH (a:Argument)
                WHERE a.stance = $stance AND a.discussion_id = $discussion_id
                RETURN a.text AS text
            """, stance=stance, discussion_id=discussion_id)

            arguments = [record["text"] for record in result if record["text"].strip()]
            if not arguments:
                continue

            # Build prompt input
            argument_block = "\n".join([f"- {arg}" for arg in arguments])
            response = argument_grouping_chain.invoke({"arguments": argument_block, "stance": stance})
            content = response.content.strip()

            # Parse and process results
            current_group = None
            group_map = {}
            for line in content.splitlines():
                if line.strip().startswith("Group:"):
                    current_group = line.replace("Group:", "").strip()
                    group_map[current_group] = []
                elif line.strip().startswith("-") and current_group:
                    arg = line.strip().lstrip("-").strip()
                    group_map[current_group].append(arg)

            # Write group and links to database
            for group_summary, arg_list in group_map.items():
                # Create unique group identifier combining summary, stance, and discussion_id
                session.run("""
                    MERGE (g:ArgumentGroup {summary: $summary, stance: $stance, discussion_id: $discussion_id})
                    SET g.updated_at = datetime()
                """, summary=group_summary, stance=stance, discussion_id=discussion_id)

                for arg in arg_list:
                    session.run("""
                        MATCH (a:Argument {text: $text, discussion_id: $discussion_id})
                        MATCH (g:ArgumentGroup {summary: $summary, stance: $stance, discussion_id: $discussion_id})
                        MERGE (a)-[:HAS_GROUP]->(g)
                    """, text=arg, discussion_id=discussion_id, summary=group_summary, stance=stance)


# Cypher helpers
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
    record = result.single()
    return record["exists"] if record else False

def check_reply_exists(tx, reply_id, discussion_id):
    result = tx.run("""
        MATCH (r:Reply {id: $reply_id, discussion_id: $discussion_id})
        RETURN count(r) > 0 AS exists
    """, reply_id=reply_id, discussion_id=discussion_id)
    record = result.single()
    return record["exists"] if record else False

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
        SET r.body = $body, 
            r.author = $author, 
            r.score = $score, 
            r.parent_comment_id = $parent_comment_id, 
            r.updated_at = datetime()
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