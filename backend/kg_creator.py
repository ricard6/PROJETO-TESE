from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from neo4j import GraphDatabase
import os
import json
from dotenv import load_dotenv
from typing import Any, Dict
import datetime

load_dotenv()

# ---- Configs ----
openai_api_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# ---- Neo4j Driver ----
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

class KGRequest(BaseModel):
    thread_data: Dict[str, Any]

# ---- LangChain LLM ----
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=openai_api_key
)

# ---- Prompt Template ----
kg_prompt = PromptTemplate(
    input_variables=["title", "selftext", "comments"],
    template="""
        You are an AI trained to extract argument knowledge graphs from Reddit discussions.

        Given a Reddit post and its comments and replies, extract meaningful arguments, their stance (FOR, AGAINST, NEUTRAL), and argument relationships (e.g., replies that support, disagree, elaborate, or question).

        Format the output as a JSON list of triples. Do not include any Markdown code block markers or extra text in your response, just the raw JSON.
        Example format: [{{"source": "<text>", "relation": "<relation>", "target": "<text>", "stance": "FOR/AGAINST/NEUTRAL"}}, ...]

        Reddit Post Title: {title}
        Reddit Post Content: {selftext}

        Top Comments and Replies:
        {comments}

        Extracted Triples:
        """
)

kg_chain = LLMChain(llm=llm, prompt=kg_prompt)

# ---- Pydantic Model for Input ----
class RedditThread(BaseModel):
    title: str
    selftext: str
    comments: list  # Should be full list of top comments + replies

# ---- Helper: Format comments for prompt ----
def format_comments(comments: list) -> str:
    formatted = []
    for c in comments:
        formatted.append(f"Comment by {c['author']} ({c['score']} pts): {c['body']}")
        for reply in c.get("replies", []):
            formatted.append(f"↳ Reply by {reply['author']} ({reply['score']} pts): {reply['body']}")
    return "\n\n".join(formatted)

# ---- Neo4j Insertion ----
def insert_triples_to_neo4j(triples: list, thread_title: str):
    """
    Insert knowledge graph triples into Neo4j with improved structure:
    - Central Topic node
    - Different node types for different stances
    - Direct relationship types
    """
    with driver.session() as session:
        tx = session.begin_transaction()
        try:
            # Create the central Topic node
            tx.run(
                """
                MERGE (t:Topic {title: $title})
                """,
                title=thread_title
            )
            
            # Process each triple
            for triple in triples:
                source_text = triple["source"]
                target_text = triple["target"]
                relation = triple["relation"].upper()  # Uppercase for Neo4j relationship type
                stance = triple["stance"]
                
                # Create source node with stance
                if stance == "FOR":
                    tx.run(
                        """
                        MERGE (a:Argument:Supporting {text: $source, stance: $stance})
                        """,
                        source=source_text,
                        stance=stance
                    )
                elif stance == "AGAINST":
                    tx.run(
                        """
                        MERGE (a:Argument:Opposing {text: $source, stance: $stance})
                        """,
                        source=source_text,
                        stance=stance
                    )
                else:
                    tx.run(
                        """
                        MERGE (a:Argument:Neutral {text: $source, stance: $stance})
                        """,
                        source=source_text,
                        stance=stance
                    )
                
                # Create target node and connect to topic
                tx.run(
                    """
                    MERGE (b:Argument {text: $target})
                    WITH b
                    MATCH (t:Topic {title: $title})
                    MERGE (b)-[:RELATES_TO]->(t)
                    """,
                    target=target_text,
                    title=thread_title
                )
                
                # Create typed relationship based on relation
                if relation == "SUPPORTS":
                    tx.run(
                        """
                        MATCH (a:Argument {text: $source})
                        MATCH (b:Argument {text: $target})
                        MERGE (a)-[:SUPPORTS {stance: $stance}]->(b)
                        """,
                        source=source_text,
                        target=target_text,
                        stance=stance
                    )
                elif relation == "DISAGREES":
                    tx.run(
                        """
                        MATCH (a:Argument {text: $source})
                        MATCH (b:Argument {text: $target})
                        MERGE (a)-[:DISAGREES {stance: $stance}]->(b)
                        """,
                        source=source_text,
                        target=target_text,
                        stance=stance
                    )
                elif relation == "ELABORATES":
                    tx.run(
                        """
                        MATCH (a:Argument {text: $source})
                        MATCH (b:Argument {text: $target})
                        MERGE (a)-[:ELABORATES {stance: $stance}]->(b)
                        """,
                        source=source_text,
                        target=target_text,
                        stance=stance
                    )
                else:
                    # Default relationship
                    tx.run(
                        """
                        MATCH (a:Argument {text: $source})
                        MATCH (b:Argument {text: $target})
                        MERGE (a)-[:RELATES {type: $relation, stance: $stance}]->(b)
                        """,
                        source=source_text,
                        target=target_text,
                        relation=relation,
                        stance=stance
                    )
            
            tx.commit()
            print(f"✅ Successfully inserted topic and {len(triples)} argument relationships into Neo4j")
        except Exception as e:
            tx.rollback()
            print(f"❌ Failed to insert knowledge graph: {str(e)}")
            raise e

# ---- Main Entry Function ----
def process_reddit_thread(thread_data: dict):
    comments_text = format_comments(thread_data["comments"])

    response = kg_chain.run(
        title=thread_data["post"]["title"],
        selftext=thread_data["post"]["selftext"],
        comments=comments_text
    )

    try:
        extracted_triples = json.loads(response)
        insert_triples_to_neo4j(extracted_triples)
        print(f"✅ Successfully inserted {len(extracted_triples)} triples into the knowledge graph.")
    except Exception as e:
        print("❌ Failed to parse or insert triples:", str(e))

def create_knowledge_graph(thread_data: dict) -> dict:
    """
    Main callable function to trigger KG extraction and storage from Reddit thread data.
    """
    comments_text = format_comments(thread_data["comments"])
    thread_title = thread_data["post"]["title"]  # Get the thread title

    # Generate response using LLM
    response = kg_chain.run(
        title=thread_title,
        selftext=thread_data["post"]["selftext"],
        comments=comments_text
    )
    
    try:
        # Extract JSON from code blocks if present
        if "```json" in response and "```" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
            
        extracted_triples = json.loads(json_str)
        
        # Pass the thread title to the insertion function
        insert_triples_to_neo4j(extracted_triples, thread_title)
        
        return {
            "status": "success",
            "inserted_triples": len(extracted_triples)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "raw_response": response
        }

