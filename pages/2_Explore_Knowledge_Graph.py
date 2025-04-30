import streamlit as st
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load Neo4j credentials
load_dotenv()
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Run a Cypher query
def run_query(cypher, parameters={}):
    with driver.session() as session:
        results = session.run(cypher, parameters)
        return [record.data() for record in results]

# Streamlit UI
st.set_page_config(page_title="🧠 Explore Discussions", layout="wide")
st.title("🧠 Explore Argument Graph")

# Define predefined queries
query_options = {
    "All Topics": {
        "query": "MATCH (t:Topic) RETURN t.title AS Title, t.url AS URL"
    },
    "Arguments by Topic and Stance": {
        "query": """
            MATCH (c)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t:Topic {title: $title})
            WITH c
            MATCH (a:Argument)-[:EXTRACTED_FROM]->(c)
            RETURN a.text AS Argument, a.stance AS Stance
        """,
        "requires_input": True,
        "input_label": "Select Topic Title"
    },
    "Count Arguments by Stance": {
        "query": """
            MATCH (c)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t:Topic {title: $title})
            WITH c
            MATCH (a:Argument)-[:EXTRACTED_FROM]->(c)
            RETURN a.stance AS Stance, count(*) AS Count
        """,
        "requires_input": True,
        "input_label": "Select Topic Title"
    }
}

selected_query_label = st.selectbox("Choose a Query to Run", list(query_options.keys()))
selected_query = query_options[selected_query_label]

parameters = {}

if selected_query.get("requires_input"):
    # Fetch available topic titles from Neo4j
    topic_titles_result = run_query("MATCH (t:Topic) RETURN t.title AS title ORDER BY t.title")
    topic_titles = [row["title"] for row in topic_titles_result]

    if topic_titles:
        selected_topic = st.selectbox(selected_query["input_label"], topic_titles)
        parameters = {"title": selected_topic}
    else:
        st.warning("No topics available in the database.")
        st.stop()

if st.button("Run Query"):
    results = run_query(selected_query["query"], parameters)
    if results:
        st.write(f"### Results:")
        st.dataframe(results, use_container_width=True)
    else:
        st.info("No results found.")
