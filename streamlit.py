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
        sorted_comments = thread_data['comments']

        # Classify and group comments
        grouped_comments = {"FOR": [], "AGAINST": [], "NEUTRAL": []}

        for comment in sorted_comments:
            # Extract top 5 replies for each comment
            sorted_replies = sorted(comment['replies'], key=lambda x: x['score'], reverse=True)[:5]

            # Add parent context if available
            parent_context = f"Parent Comment: {comment.get('parent_body', 'N/A')}\n\n" if 'parent_body' in comment else ""

            # Classify stance
            with st.spinner(f"Classifying stance..."):
                stance_response = requests.post(
                    stanceClassifier_endpoint,
                    json={
                        "thread_title": thread_data['post']['title'],
                        "thread_selftext": thread_data['post']['selftext'],
                        "identified_topic": topicIdentifier,
                        "comment_body": f"{parent_context}{comment['body']}"
                    }
                )

            if stance_response.status_code == 200:
                stance_result = stance_response.json().get("stance", "NEUTRAL")
                grouped_comments[stance_result].append({
                    "author": comment["author"],
                    "score": comment["score"],
                    "body": comment["body"],
                    "replies": sorted_replies  # Include top 5 replies
                })

        # Summarize arguments per stance
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

            stance_summaries = stance_summary_response.json().get("summaries", {}) if stance_summary_response.status_code == 200 else {
                "FOR": "Unable to summarize favorable arguments.",
                "AGAINST": "Unable to summarize opposing arguments.",
                "NEUTRAL": "Unable to summarize neutral arguments."
            }
        
        # Sample data (Replace with actual stance counts)
        stance_counts = {
            "FOR": len(grouped_comments["FOR"]),
            "AGAINST": len(grouped_comments["AGAINST"]),
            "NEUTRAL": len(grouped_comments["NEUTRAL"])
        }

        total_comments = sum(stance_counts.values())  # Total number of classified comments

        # Calculate stance percentages
        if total_comments > 0:
            stance_percentages = {
                "FOR": (stance_counts["FOR"] / total_comments) * 100,
                "AGAINST": (stance_counts["AGAINST"] / total_comments) * 100,
                "NEUTRAL": (stance_counts["NEUTRAL"] / total_comments) * 100
            }
        else:
            stance_percentages = {"FOR": 0, "AGAINST": 0, "NEUTRAL": 0}

        # Convert percentages to CSS-friendly format
        bar_style = f"""
            <div style="display: flex; width: 100%; height: 25px; 
                        border-radius: 15px; overflow: hidden; 
                        border: 2px solid #ddd; margin-bottom: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
                <div style="width: {stance_percentages['FOR']}%; background: linear-gradient(to right, #2ecc71, #27ae60);"></div>
                <div style="width: {stance_percentages['NEUTRAL']}%; background: linear-gradient(to right, #f1c40f, #f39c12);"></div>
                <div style="width: {stance_percentages['AGAINST']}%; background: linear-gradient(to right, #e74c3c, #c0392b);"></div>
            </div>
        """
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        st.subheader("📊 Stance Distribution")
        st.markdown(bar_style, unsafe_allow_html=True)

        # Show exact percentage values
        legend_html = f"""
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 15px; height: 15px; background-color: #2ecc71; border-radius: 3px; margin-right: 5px;"></div>
                    <span><b>For:</b> {stance_percentages['FOR']:.0f}%</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 15px; height: 15px; background-color: #f1c40f; border-radius: 3px; margin-right: 5px;"></div>
                    <span><b>Neutral:</b> {stance_percentages['NEUTRAL']:.0f}%</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 15px; height: 15px; background-color: #e74c3c; border-radius: 3px; margin-right: 5px;"></div>
                    <span><b>Against:</b> {stance_percentages['AGAINST']:.0f}%</span>
                </div>
            </div>
        """

        st.markdown(legend_html, unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        # Display grouped comments and their summaries
        st.subheader("Arguments by Stance")

        # Column containers for the summaries
        col1, col2 = st.columns([1.5, 1.5])

        with col1:
            st.markdown("### 🟩 Favorable Arguments")
            st.markdown(stance_summaries["FOR"])

        with col2:
            st.markdown("### 🟥 Against Arguments")
            st.markdown(stance_summaries["AGAINST"])

        # New row for the expanders - this ensures they're aligned
        expander_col1, expander_col2 = st.columns([1.5, 1.5])

        with expander_col1:
            with st.expander("Original Comments", expanded=False):
                for comment in grouped_comments["FOR"]:
                    st.write(f"**{comment['author']}** ({comment['score']} points)")
                    st.write(comment["body"])
                    # Show replies
                    for reply in comment["replies"]:
                        st.markdown(f"↳ **{reply['author']}** ({reply['score']} points)")
                        formatted_reply = "\n> ".join(reply["body"].split("\n"))
                        st.markdown(f"> {formatted_reply}")
                    st.write("---")

        with expander_col2:
            with st.expander("Original Comments", expanded=False):
                for comment in grouped_comments["AGAINST"]:
                    st.write(f"**{comment['author']}** ({comment['score']} points)")
                    st.write(comment["body"])
                    # Show replies
                    for reply in comment["replies"]:
                        st.markdown(f"↳ **{reply['author']}** ({reply['score']} points)")
                        formatted_reply = "\n> ".join(reply["body"].split("\n"))
                        st.markdown(f"> {formatted_reply}")
                    st.write("---")

        # Show neutral arguments separately
        if grouped_comments["NEUTRAL"]:
            with st.expander("🟨 Neutral Arguments", expanded=False):
                st.markdown(stance_summaries["NEUTRAL"])
                st.write("---")
                for comment in grouped_comments["NEUTRAL"]:
                    st.write(f"**{comment['author']}** ({comment['score']} points)")
                    st.write(comment["body"])
                    # Show replies
                    for reply in comment["replies"]:
                        st.markdown(f"↳ **{reply['author']}** ({reply['score']} points)")
                        formatted_reply = "\n> ".join(reply["body"].split("\n"))
                        st.markdown(f"> {formatted_reply}")
                    st.write("---")

        
