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
st.title("🧠 Explore Stored Discussions")

# First, fetch all available topic titles from Neo4j
topic_titles_result = run_query("MATCH (t:Topic) RETURN t.title AS title ORDER BY t.title")
topic_titles = [row["title"] for row in topic_titles_result]

# Topic selection first
if not topic_titles:
    st.warning("No topics available in the database.")
    st.stop()

selected_topic = st.selectbox("Choose a Topic to Explore", topic_titles)

# After topic selection, display metrics
if selected_topic:
    # Get counts by stance for this topic using the working query
    stance_counts = run_query("""
        MATCH (c)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t:Topic {title: $title})
        WITH c
        MATCH (a:Argument)-[:EXTRACTED_FROM]->(c)
        RETURN a.stance AS Stance, count(*) AS Count
    """, {"title": selected_topic})
    
    # Initialize count variables
    supporting_count = 0
    opposing_count = 0
    neutral_count = 0
    
    # Process the query results
    for item in stance_counts:
        stance = item["Stance"]
        count = item["Count"]
        
        if stance:
            stance = stance.lower()
            if "for" in stance:  # Check for "FOR"
                supporting_count = count
            elif "against" in stance:  # Check for "AGAINST"
                opposing_count = count
            elif "neutral" in stance:  # Check for "NEUTRAL"
                neutral_count = count

    # Ensure all variables are set to a default value even if no counts are found
    supporting_count = supporting_count or 0
    opposing_count = opposing_count or 0
    neutral_count = neutral_count or 0

    st.markdown("""
    <style>
        .space {
            margin-top: 30px;
        }
    </style>
    <div class="space"></div>
    """, unsafe_allow_html=True)

    # Display the metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Supporting Arguments Identified", value=supporting_count)
    with col2:
        st.metric(label="Opposing Arguments Identified", value=opposing_count)
    with col3:
        st.metric(label="Neutral Arguments Identified", value=neutral_count)
    
    # Display divider
    st.divider()
    
    # Define queries that can be applied to the selected topic
    query_options = {
        "List of Arguments by Stance": {
            "query": """
                MATCH (c)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t:Topic {title: $title})
                WITH c
                MATCH (a:Argument)-[:EXTRACTED_FROM]->(c)
                RETURN a.text AS Argument, a.stance AS Stance
            """
        },
        "Topic Details": {
            "query": """
                MATCH (t:Topic {title: $title})
                RETURN t.title AS Title, t.url AS URL
            """
        }
    }

    # After topic selection, allow query selection
    selected_query_label = st.selectbox("Choose a Query to Run", list(query_options.keys()))
    selected_query = query_options[selected_query_label]

    # Parameters already set based on topic selection
    parameters = {"title": selected_topic}

    if st.button("Run Query"):
        results = run_query(selected_query["query"], parameters)
        if results:
            st.write(f"### Results for '{selected_topic}':")
            st.dataframe(results, use_container_width=True)
        else:
            st.info(f"No results found for '{selected_topic}' with the selected query.")