import streamlit as st
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import pandas as pd

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
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# First, fetch all available topic titles from Neo4j
topic_titles_result = run_query("MATCH (t:Topic) RETURN t.title AS title ORDER BY t.title")
topic_titles = [row["title"] for row in topic_titles_result]

# Topic selection first
if not topic_titles:
    st.warning("No topics available in the database.")
    st.stop()

selected_topic = st.selectbox("Choose a topic from the discussions stored in the Discussion Database to explore its details.", topic_titles)
st.markdown(f"### {selected_topic}")

# After topic selection, display metrics
if selected_topic:
    # Get counts by stance for this topic using the working query
    stance_counts = run_query("""
        MATCH (t:Topic {title: $title})
        MATCH (a:Argument)-[:EXTRACTED_FROM]->(n)
        WHERE (n:Comment)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t)
        OR (n:Reply)-[:REPLY_TO]->(:Comment)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t)
        RETURN a.stance AS Stance, count(*) AS Count
    """, 
    {"title": selected_topic})
    
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

    st.markdown("""<style>.space {margin-top: 30px;}</style><div class="space"></div>""", unsafe_allow_html=True)

    # Display the metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Supporting Arguments Identified", value=supporting_count)
    with col2: 
        st.metric(label="Neutral Arguments Identified", value=neutral_count)
    with col3:
        st.metric(label="Opposing Arguments Identified", value=opposing_count)
    
    # Display divider
    st.divider()
    st.header("Explore each post and its extracted arguments")

    # Fetch all comments and replies related to the selected topic
    discussion_posts = run_query("""
        MATCH (parent_comment)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t:Topic {title: $title})
        OPTIONAL MATCH (reply)-[:REPLY_TO]->(parent_comment)
        WITH collect(parent_comment) + collect(reply) AS all_comments
        UNWIND all_comments AS c
        WITH DISTINCT c WHERE c IS NOT NULL
        RETURN id(c) AS comment_id, c.body AS text
        ORDER BY c.body
    """, {"title": selected_topic})

    # Build dropdown of posts (including replies)
    if discussion_posts:
        post_options = {
            f"Post #{row['comment_id']} - {row['text'][:100]}..." if row.get("text") else f"Post #{row['comment_id']} - [No text]": row["comment_id"]
            for row in discussion_posts
        }

        selected_post_label = st.selectbox("Select a Discussion Post or Reply", list(post_options.keys()))
        selected_post_id = post_options[selected_post_label]

        # Show full post/reply content
        post_details = run_query("""
            MATCH (c)
            WHERE id(c) = $comment_id
            RETURN c.body AS full_text
        """, {"comment_id": selected_post_id})

        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("### 📝 Full Post Content")
        st.write(post_details[0]["full_text"] if post_details else "No content found.")

        # Show arguments for this comment/reply
        extracted_arguments = run_query("""
            MATCH (a:Argument)-[:EXTRACTED_FROM]->(c)
            WHERE id(c) = $comment_id
            RETURN a.text AS ArgumentText, a.stance AS Stance
        """, {"comment_id": selected_post_id})

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("### 🧩 Extracted Arguments")
        if extracted_arguments:
            for arg in extracted_arguments:
                st.markdown(f"- **{arg['Stance']}**: {arg['ArgumentText']}")
        else:
            st.info("No arguments were extracted from this post or reply.")
    else:
        st.info("No posts or replies found for this topic.")


    st.divider()
    st.header("Explore Further")

    # Define queries that can be applied to the selected topic
    query_options = {
        "List of Arguments by Stance": {
            "query": """
                MATCH (t:Topic {title: $title})
                MATCH (a:Argument)-[:EXTRACTED_FROM]->(n)
                WHERE (n:Comment)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t)
                OR (n:Reply)-[:REPLY_TO]->(:Comment)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t)
                RETURN a.text AS Argument, a.stance AS Stance
            """
        },
        "Argument Groups by Popularity": {
            "query": """
                MATCH (t:Topic {title: $title})
                MATCH (a:Argument)-[:EXTRACTED_FROM]->(n)
                WHERE (n:Comment)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t)
                OR (n:Reply)-[:REPLY_TO]->(:Comment)-[:SUPPORTS|OPPOSES|NEUTRAL]->(t)
                MATCH (a)-[:HAS_GROUP]->(g:ArgumentGroup)
                RETURN 
                g.summary AS GroupSummary,
                a.text AS ArgumentText,
                a.stance AS Stance
            """
        },
        "Replies to Supporting Comments": {
            "query": """
                MATCH (c:Comment)-[:SUPPORTS]->(t:Topic {title: $title})
                MATCH (r:Reply)-[:REPLY_TO]->(c)
                RETURN c.body AS ParentComment, 
                       collect(r.body) AS Replies,
                       size(collect(r.body)) AS ReplyCount
                ORDER BY ReplyCount DESC
                LIMIT 10
            """
        },
        "Replies to Opposing Comments": {
            "query": """
                MATCH (c:Comment)-[:OPPOSES]->(t:Topic {title: $title})
                MATCH (r:Reply)-[:REPLY_TO]->(c)
                RETURN c.body AS ParentComment, 
                       collect(r.body) AS Replies,
                       size(collect(r.body)) AS ReplyCount
                ORDER BY ReplyCount DESC
                LIMIT 10
            """
        },
        "Popular Comments (by score)": {
            "query": """
                MATCH (c:Comment)-[r:SUPPORTS|OPPOSES|NEUTRAL]->(t:Topic {title: $title})
                RETURN c.body AS Comment, 
                       c.score AS Score, type(r) AS Stance
                ORDER BY c.score DESC
                LIMIT 10
            """
        },
        "Discussion url": {
            "query": """
                MATCH (t:Topic {title: $title})
                RETURN t.title AS Title, t.url AS URL
            """
        }
    }

    # After topic selection, allow query selection
    selected_query_label = st.selectbox("Choose a Query to Run", list(query_options.keys()))


    # Parameters already set based on topic selection
    parameters = {"title": selected_topic}

    # Only show display mode option if the selected query is the one you want
    if selected_query_label == "Argument Groups by Popularity":
        display_mode = st.radio("Display Mode", ["Group Overview", "Full Detail (show every argument)"])
    else:
        display_mode = None

    if st.button("Run Query"):
        results = run_query(query_options[selected_query_label]["query"], parameters)

        if results:
            st.write(f"### Results for '{selected_topic}':")

            if selected_query_label == "Argument Groups by Popularity":
                if display_mode == "Group Overview":
                    overview_data = {}
                    for row in results:
                        key = (row.get("GroupSummary"), row.get("Stance"))
                        overview_data[key] = overview_data.get(key, 0) + 1

                    overview_rows = [
                        {"Group Summary": k[0], "Stance": k[1], "Argument Count": v}
                        for k, v in overview_data.items()
                    ]
                    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

                elif display_mode == "Full Detail (show every argument)":
                    detail_rows = [
                        {
                            "Group Summary": row.get("GroupSummary", ""),
                            "Argument Text": row.get("ArgumentText", ""),
                            "Stance": row.get("Stance", "")
                        }
                        for row in results if row.get("ArgumentText")
                    ]

                    if detail_rows:
                        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)
                    else:
                        st.info("No arguments found to display.")

            elif selected_query_label == "Replies to Supporting Comments" or "Replies to Opposing Comments":
                if "Replies" in results[0] and "ParentComment" in results[0]:
                    for i, item in enumerate(results, 1):
                        with st.expander(f"Comment {i}"):
                            st.markdown("**🧠 Parent Comment:**")
                            st.markdown(item["ParentComment"])

                            replies = item.get("Replies", [])
                            if replies:
                                st.markdown("**💬 Replies:**")
                                for j, reply in enumerate(replies, 1):
                                    st.markdown(f"**Reply #{j}:**\n{reply}")
                                    st.divider()
                            else:
                                st.write("No replies found.")

            else:
                # Fallback for other query types
                st.dataframe(results, use_container_width=True)

        else:
            st.info(f"No results found for '{selected_topic}' with the selected query.")





