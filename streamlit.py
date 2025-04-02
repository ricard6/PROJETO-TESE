import streamlit as st
import requests

# Streamlit app title
st.title("Discussion Analysis")

#App description
st.write("This tool will summarize a Reddit discussion, extracting and displaying the best arguments used in favour, or against the topic of discussion.")

#Reddit link input textbox
reddit_url = st.text_input("Enter Reddit thread URL:", placeholder="https://www.reddit.com/r/subreddit/comments/...")

# Backend URLs
backend_url = "http://127.0.0.1:8000"
summarizer_endpoint = f"{backend_url}/summarizer"
reddit_scraper_endpoint = f"{backend_url}/reddit_scraper" 
topicIdentifier_endpoint = f"{backend_url}/topicIdentifier"
stanceClassifier_endpoint = f"{backend_url}/stanceClassifier"

# Button to process
if st.button("Go!", type="primary"):
    # First scrape the Reddit thread
    with st.spinner("Initializing analysis..."):
        scrape_response = requests.post(
            reddit_scraper_endpoint,
            json={"url": reddit_url}
        )

    if scrape_response.status_code == 200:
        thread_data = scrape_response.json().get("thread_data")

        # Identify topic discussion using LLM
        with st.spinner("Identifying discussion topic..."):
            full_text = f"{thread_data['post']['title']} {thread_data['post']['selftext']}"
            topicIdentifier_response = requests.post(
                topicIdentifier_endpoint,
                json={"text": full_text} 
            )

        if topicIdentifier_response.status_code == 200:
            topicIdentifier = topicIdentifier_response.json().get("topic")
        else:
            st.error(f"Failed to identify the discussion topic: {topicIdentifier_response.text}")
            topicIdentifier = "Unidentified Topic"

        # Display basic thread info
        st.header(topicIdentifier)
        st.write(f"**Subreddit:** r/{thread_data['post']['subreddit']}")
        st.write(f"**Number of Upvotes:** {thread_data['post']['score']}")
        st.write(f"**Number of Comments:** {thread_data['post']['num_comments']}")

        # Display post content
        with st.expander("View Original Post Content", expanded=False):
            st.write(thread_data['post']['selftext'])

        # Sort comments by score (highest first)
        sorted_comments = sorted(
            [c for c in thread_data['comments'] if c['author'] != "AutoModerator"], 
            key=lambda x: x['score'], reverse=True
        )

        # Extract top 10 comments
        top_comments = [comment['body'] for comment in sorted_comments[:10]]

        # Classify comments and group them
        grouped_comments = {"FOR": [], "AGAINST": [], "NEUTRAL": []}

        for comment in sorted_comments[:10]:  
            with st.spinner(f"Classifying stance..."):
                stance_response = requests.post(
                    stanceClassifier_endpoint,
                    json={
                        "thread_title": thread_data['post']['title'],
                        "thread_selftext": thread_data['post']['selftext'],
                        "identified_topic": topicIdentifier,
                        "comment_body": comment['body']
                    }
                )

            if stance_response.status_code == 200:
                stance_result = stance_response.json().get("stance", "NEUTRAL")
                grouped_comments[stance_result].append({
                    "author": comment["author"],
                    "score": comment["score"],
                    "body": comment["body"]
                })

        # Get stance summaries
        with st.spinner("Analyzing arguments by stance..."):
            stance_summary_response = requests.post(
                summarizer_endpoint,
                json={
                    "grouped_comments": {
                        "FOR": [c["body"] for c in grouped_comments["FOR"]],
                        "AGAINST": [c["body"] for c in grouped_comments["AGAINST"]],
                        "NEUTRAL": [c["body"] for c in grouped_comments["NEUTRAL"]]
                    }
                }
            )

            if stance_summary_response.status_code == 200:
                stance_summaries = stance_summary_response.json().get("summaries", {})
            else:
                st.error("Failed to summarize arguments by stance.")
                stance_summaries = {
                    "FOR": "Unable to summarize favorable arguments.",
                    "AGAINST": "Unable to summarize opposing arguments.",
                    "NEUTRAL": "Unable to summarize neutral arguments."
                }
                

        # Display grouped comments and their summaries
        st.subheader("Arguments by Stance")
        col1, col2 = st.columns([1.5, 1.5])

        with col1:
            st.markdown("### 🟩 Favorable Arguments")
            st.markdown(stance_summaries["FOR"])
            with st.expander("Original Comments", expanded=False):
                for comment in grouped_comments["FOR"]:
                    st.write(f"**{comment['author']}** ({comment['score']} points)")
                    st.write(comment["body"])
                    st.write("---")

        with col2:
            st.markdown("### 🟥 Against Arguments")
            st.markdown(stance_summaries["AGAINST"])
            with st.expander("Original Comments", expanded=False):
                for comment in grouped_comments["AGAINST"]:
                    st.write(f"**{comment['author']}** ({comment['score']} points)")
                    st.write(comment["body"])
                    st.write("---")

        # Show neutral arguments separately
        if grouped_comments["NEUTRAL"]:
            with st.expander("🟡 Neutral Arguments", expanded=False):
                st.markdown(stance_summaries["NEUTRAL"])
                st.write("---")
                for comment in grouped_comments["NEUTRAL"]:
                    st.write("**Original Comments**")
                    st.write(f"**{comment['author']}** ({comment['score']} points)")
                    st.write(comment["body"])
                    st.write("---")
        else:
            st.error(f"Failed to fetch Reddit thread: {scrape_response.json().get('error', 'Unknown error')}")
