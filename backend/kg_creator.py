from neo4j import GraphDatabase
import os
import uuid
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Any, Dict, List
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

    Your task is to extract only arguments that can be used in scientific, academic, or logical contexts. Arguments should clearly support or oppose a specific claim related to the discussion topic. Avoid insults, and vague statements.

    ### INSTRUCTIONS:
    - Extract arguments that include **reasoning, causal relationships, or factual claims**.
    - Each argument must be **self-contained**: don't use pronouns like “this” or “it” without defining them.
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

argument_grouping_prompt = PromptTemplate(
    input_variables=["stance", "arguments"],
    template="""
    You are an AI system trained to group semantically similar arguments from Reddit discussions.

    ### Objective:
    Given a list of arguments that all share the same stance ("{stance}") toward a topic, group those that express the same core idea — even if worded differently. For each group, write a single, clear summary that captures what all grouped arguments are trying to say.

    ### Instructions:
    - All arguments are either FOR, AGAINST, or NEUTRAL with respect to the topic.
    - Only group arguments that express the same reasoning or point.
    - Do NOT merge arguments that express different ideas, even if they are related.
    - Use the exact original argument wording — do not rewrite or paraphrase.
    - If an argument doesn’t belong in any group, place it in its own group.

    ### Format:

    Group: <summary of shared idea>
    - <original argument 1>
    - <original argument 2>

    Group: <another shared idea>
    - <original argument 3>

    ### Arguments:
    {arguments}
    """
)

argument_grouping_chain = argument_grouping_prompt | llm

# Pydantic model
class KGRequest(BaseModel):
    thread_data: Dict[str, Any]

# Argument extraction
def extract_arguments(text: str, topic: str) -> List[str]:
    response = argument_chain.invoke({"text": text, "topic": topic})
    content = response.content.strip()  # ✅ get actual string

    if "No clear arguments found" in content:
        return []
    return [line.split(" ", 1)[-1].strip() for line in content.split('\n') if line.strip()]

# Knowledge graph creation
def create_knowledge_graph(thread_data: dict) -> dict:
    # Get post title and URL (used as unique discussion ID)
    topic_title = thread_data['post'].get('title', 'Unknown Topic')
    thread_url = thread_data['post'].get('url', 'unknown')
    comment_count = 0
    reply_count = 0
    argument_count = 0

    with driver.session() as session:
        # Check if the thread already exists in the graph
        existing = session.execute_read(check_existing_discussion_by_url, thread_url)
        # Ensure the topic node is present in the graph
        discussion_id = existing if existing else str(uuid.uuid4())
        session.execute_write(merge_topic, topic_title, discussion_id, thread_url)

        # Loop through each stance category: FOR, AGAINST, NEUTRAL
        for stance in ["FOR", "AGAINST", "NEUTRAL"]:
            for i, comment in enumerate(thread_data["classified_comments"].get(stance, [])):
                # Ensure each comment has a unique ID
                if "id" not in comment or not comment["id"]:
                    comment["id"] = f"comment_{stance}_{i}"
                comment.update({"discussion_id": discussion_id})

                # Check if the comment is already in the database
                comment_exists = session.execute_read(check_comment_exists, comment["id"], discussion_id)
                session.execute_write(merge_comment, comment)
                
                # Only process new comments
                if not comment_exists:
                    comment_count += 1
                    session.execute_write(connect_comment_to_topic, comment["id"], topic_title, stance, discussion_id)

                    # Extract arguments from the comment and store them in the graph
                    arguments = extract_arguments(comment.get("body", ""), topic_title)
                    for arg in arguments:
                        arg = arg.strip()
                        session.execute_write(merge_argument, arg, stance, discussion_id)
                        session.execute_write(connect_argument_to_comment, arg, comment["id"], discussion_id)
                        argument_count += 1
                
                # Handle replies for the comment
                for j, reply in enumerate(comment.get("replies", [])):
                    if "id" not in reply or not reply["id"]:
                        reply["id"] = f"reply_{comment['id']}_{j}"
                    else:
                        reply["id"] = f"{comment['id']}_{reply['id']}"
                    reply.update({"discussion_id": discussion_id})

                    # Check if the reply is already in the database
                    reply_exists = session.execute_read(check_reply_exists, reply["id"], discussion_id)
                    session.execute_write(merge_reply, reply)

                    if not reply_exists:
                        reply_count += 1
                        session.execute_write(connect_reply_to_comment, reply["id"], comment["id"], discussion_id)

                        # Extract and add arguments from the reply
                        arguments = extract_arguments(reply.get("body", ""), topic_title)
                        for arg in arguments:
                            arg = arg.strip()
                            session.execute_write(merge_argument, arg, reply.get("stance", "NEUTRAL"), discussion_id)
                            session.execute_write(connect_argument_to_reply, arg, reply["id"], discussion_id)
                            argument_count += 1

        group_arguments_by_stance(discussion_id)
    

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


def group_arguments_by_stance(discussion_id: str):
    if not argument_grouping_chain or not driver:
        return

    with driver.session() as session:
        # Loop through each stance
        for stance in ["FOR", "AGAINST", "NEUTRAL"]:
            # Get all arguments for this stance
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
                session.run("""
                    MERGE (g:ArgumentGroup {summary: $summary, stance: $stance})
                    SET g.updated_at = datetime()
                """, summary=group_summary, stance=stance)

                for arg in arg_list:
                    session.run("""
                        MATCH (a:Argument {text: $text, discussion_id: $discussion_id})
                        MATCH (g:ArgumentGroup {summary: $summary, stance: $stance})
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
