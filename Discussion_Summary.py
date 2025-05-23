import streamlit as st
import requests

# App config
st.set_page_config(
        page_title="Discussion Analysis", layout = "wide")


# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if 'history' not in st.session_state:
    st.session_state.history = []

if "discussions" not in st.session_state:
    st.session_state.discussions = []

if "selected_discussion" not in st.session_state:
    st.session_state.selected_discussion = None

if "processing_state" not in st.session_state:
    st.session_state.processing_state = {
        "topic": None,
        "thread_data": None,
        "grouped_comments": None,
        "stance_summaries": None,
        "stance_percentages": None,
        "kg_info": None
    }

# Backend URLs
backend_url = "http://127.0.0.1:8000"
summarizer_endpoint = f"{backend_url}/summarizer"
reddit_scraper_endpoint = f"{backend_url}/reddit_scraper" 
topicIdentifier_endpoint = f"{backend_url}/topicIdentifier"
stanceClassifier_endpoint = f"{backend_url}/stanceClassifier"
kgCreator_endpoint = f"{backend_url}/kgCreator"

# Helper function to render replies
def render_replies(replies):
    for reply in replies:
        stance_emoji = {
            "FOR": "🟩",
            "AGAINST": "🟥",
            "NEUTRAL": "🟨"
        }.get(reply["stance"], "🟨")
        st.markdown(f"↳ **{reply['author']}** ({reply['score']} points) {stance_emoji}")
        formatted_reply = "\n> ".join(reply["body"].split("\n"))
        st.markdown(f"> {formatted_reply}")

# Sidebar
st.sidebar.title("Discussion Navigator")
st.sidebar.markdown("""
    <style>
        .sidebar-button-space {
            margin-top: 20px;
        }
    </style>
    <div class="sidebar-button-space"></div>
""", unsafe_allow_html=True)
if st.sidebar.button("➕ New Discussion Analysis"):
    st.session_state.page = "home"
    st.session_state.selected_discussion = None
    st.session_state.processing_state = {
        "topic": None,
        "thread_data": None,
        "grouped_comments": None,
        "stance_summaries": None,
        "stance_percentages": None,
        "kg_info": None
    }
    st.rerun()

# Apply custom CSS to ensure text is left-aligned/justified
st.markdown("""
<style>
    .stButton>button {
        text-align: left !important;
        justify-content: flex-start !important;
        white-space: normal !important;
        height: auto !important;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Display saved discussions in sidebar
for idx, discussion in enumerate(st.session_state.discussions):
    raw_topic = discussion.get("topic", "")
    
    # Strip markdown formatting (remove asterisks)
    clean_topic = raw_topic.replace("*", "")
    
    # Get the first line or the whole thing if it's a single line
    main_topic_line = clean_topic.splitlines()[0] if clean_topic else "Untitled"
    
    # Further clean up if it starts with "Main Topic:"
    topic_title = main_topic_line.split(":", 1)[1].strip() if "Main Topic:" in main_topic_line else main_topic_line
    
    # Truncate if too long
    if len(topic_title) > 60:
        topic_title = topic_title[:57] + "..."
        
    if st.sidebar.button(topic_title, key=f"discussion_{idx}"):
        st.session_state.page = "discussion_view"
        st.session_state.selected_discussion = discussion
        st.rerun()

# Home Page Input
if st.session_state.page == 'home':
    st.title("Discussion Analysis")
    st.write("This tool will summarize a Reddit discussion, extracting and displaying the best arguments used in favour, or against the topic of discussion.")
    reddit_url = st.text_input("Enter Reddit thread URL:", placeholder="https://www.reddit.com/r/subreddit/comments/...")
    
    # Check if URL already processed
    if st.button("Go!", type="primary") and reddit_url:
        # Check if we already have this URL in discussions
        existing_discussion = next((d for d in st.session_state.discussions if d.get("url") == reddit_url), None)
        if existing_discussion:
            # We've already analyzed this URL, so use cached results
            st.session_state.page = "discussion_view"
            st.session_state.selected_discussion = existing_discussion
        else:
            # New URL, set up for incremental analysis
            st.session_state.page = "incremental_analysis"
            st.session_state.reddit_url = reddit_url
        st.rerun()

# Incremental Analysis Page - shows results as they're processed
elif st.session_state.page == 'incremental_analysis':
    reddit_url = st.session_state.reddit_url
    process_state = st.session_state.processing_state
    
    # Step 1: Scrape Reddit and identify discussion topic
    if process_state["thread_data"] is None:
        with st.spinner("Initializing analysis..."):
            scrape_response = requests.post(reddit_scraper_endpoint, json={"url": reddit_url})

        if scrape_response.status_code == 200:
            thread_data = scrape_response.json().get("thread_data")
            process_state["thread_data"] = thread_data

            # Identify the discussion topic
            with st.spinner("Identifying discussion topic..."):
                full_text = f"{thread_data['post']['title']} {thread_data['post']['selftext']}"
                topicIdentifier_response = requests.post(
                    topicIdentifier_endpoint,
                    json={"text": full_text} 
                )

            if topicIdentifier_response.status_code == 200:
                topicIdentifier = topicIdentifier_response.json().get("topic")
                process_state["topic"] = topicIdentifier
            else:
                st.error(f"Failed to identify the discussion topic: {topicIdentifier_response.text}")
                topicIdentifier = "Unidentified Topic"
                process_state["topic"] = topicIdentifier

            if topicIdentifier not in [t[0] for t in st.session_state.history]:
                st.session_state.history.append((topicIdentifier, reddit_url))
        else:
            st.error(f"Failed to scrape Reddit thread: {scrape_response.text}")
            st.stop()
    
    # Display the header and basic info as soon as we have it
    thread_data = process_state["thread_data"]
    topicIdentifier = process_state["topic"]
    
    st.header(topicIdentifier)
    st.write(f"**Subreddit:** r/{thread_data['post']['subreddit']}")
    st.write(f"**Number of Upvotes:** {thread_data['post']['score']}")
    st.write(f"**Number of Comments:** {thread_data['post']['num_comments']}")

    # Original post content
    with st.expander("View Original Post Content", expanded=False):
        st.write(thread_data['post']['selftext'])
    
    # Step 2: Classify comments and stances
    if process_state["grouped_comments"] is None:
        top_comments = thread_data['comments']

        grouped_comments = {"FOR": [], "AGAINST": [], "NEUTRAL": []}
        all_classified_bodies = {"FOR": [], "AGAINST": [], "NEUTRAL": []}
        all_replies = []  # <-- Move here, outside the loop

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, comment in enumerate(top_comments):
            sorted_replies = sorted(comment['replies'], key=lambda x: x['score'], reverse=True)[:5]
            parent_context = f"Parent Comment: {comment.get('parent_body', 'N/A')}\n\n" if 'parent_body' in comment else ""

            progress = (i + 1) / len(top_comments)
            progress_bar.progress(progress)
            status_text.text(f"Classifying comment {i+1}/{len(top_comments)}")

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

                classified_replies = []
                for reply in sorted_replies:
                    reply_stance_response = requests.post(
                        stanceClassifier_endpoint,
                        json={
                            "thread_title": thread_data['post']['title'],
                            "thread_selftext": thread_data['post']['selftext'],
                            "identified_topic": topicIdentifier,
                            "comment_body": reply['body']
                        }
                    )
                    reply_stance = reply_stance_response.json().get("stance", "NEUTRAL") if reply_stance_response.status_code == 200 else "NEUTRAL"

                    classified_replies.append({
                        "author": reply["author"],
                        "score": reply["score"],
                        "body": reply["body"],
                        "stance": reply_stance
                    })

                    all_classified_bodies[reply_stance].append(reply["body"])

                # Append replies of this comment to the global all_replies list
                all_replies.extend(classified_replies)

                grouped_comments[stance_result].append({
                    "author": comment["author"],
                    "score": comment["score"],
                    "body": comment["body"],
                    "replies": classified_replies
                })

                all_classified_bodies[stance_result].append(comment["body"])

        progress_bar.empty()
        status_text.empty()

        process_state["grouped_comments"] = grouped_comments
        process_state["all_classified_bodies"] = all_classified_bodies

        # Calculate stance distribution
        stance_counts = {"FOR": 0, "AGAINST": 0, "NEUTRAL": 0}

        # Count stance from comments
        for stance in grouped_comments:
            stance_counts[stance] += len(grouped_comments[stance])

        # Count stance from **all** replies (not just last comment’s)
        for reply in all_replies:
            reply_stance = reply.get("stance")
            if reply_stance in stance_counts:
                stance_counts[reply_stance] += 1

        total = sum(stance_counts.values())
        stance_percentages = {
            stance: (count / total) * 100 if total > 0 else 0
            for stance, count in stance_counts.items()
        }

        process_state["stance_percentages"] = stance_percentages
            
    # Display stance distribution as soon as we have it
    if process_state["stance_percentages"]:
        stance_percentages = process_state["stance_percentages"]
        
        bar_style = f"""
            <div style="display: flex; width: 100%; height: 25px; 
                        border-radius: 15px; overflow: hidden; 
                        border: 2px solid #ddd; margin-bottom: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
                <div style="width: {stance_percentages['FOR']}%; background: linear-gradient(to right, #27ae60, #2ecc71);"></div>
                <div style="width: {stance_percentages['NEUTRAL']}%; background: linear-gradient(to right, #f1c40f, #f39c12);"></div>
                <div style="width: {stance_percentages['AGAINST']}%; background: linear-gradient(to right, #e74c3c, #c0392b);"></div>
            </div>
        """
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        st.divider()
        st.subheader("📊 Stance Distribution")
        st.markdown(bar_style, unsafe_allow_html=True)

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

    st.divider()
    
    # Step 3: Summarize arguments by stance
    if process_state["stance_summaries"] is None and process_state["all_classified_bodies"]:
        with st.spinner("Analyzing arguments by stance..."):
            stance_summary_response = requests.post(
                summarizer_endpoint,
                json={"grouped_comments": process_state["all_classified_bodies"]}
            )

            if stance_summary_response.status_code == 200:
                stance_summaries = stance_summary_response.json().get("summaries", {})
            else:
                stance_summaries = {
                    "FOR": "Unable to summarize favorable arguments.",
                    "AGAINST": "Unable to summarize opposing arguments.",
                    "NEUTRAL": "Unable to summarize neutral arguments."
                }
                
            process_state["stance_summaries"] = stance_summaries
    
    # Display argument summaries as soon as we have them
    if process_state["stance_summaries"]:
        stance_summaries = process_state["stance_summaries"]
        
        st.subheader("Arguments by Stance - Summary")
        col1, col2 = st.columns([1.5, 1.5])

        with col1:
            st.markdown("### 🟩 Favorable Arguments")
            st.markdown(stance_summaries["FOR"])

        with col2:
            st.markdown("### 🟥 Against Arguments")
            st.markdown(stance_summaries["AGAINST"])
            
        # Display original comments if we have them
        if process_state["grouped_comments"]:
            grouped_comments = process_state["grouped_comments"]
            
            expander_col1, expander_col2 = st.columns([1.5, 1.5])

            with expander_col1:
                with st.expander("Original Favorable Comments", expanded=False):
                    for comment in grouped_comments["FOR"]:
                        st.write(f"**{comment['author']}** ({comment['score']} points)")
                        st.write(comment["body"])
                        render_replies(comment["replies"])
                        st.write("---")

            with expander_col2:
                with st.expander("Original Opposing Comments", expanded=False):
                    for comment in grouped_comments["AGAINST"]:
                        st.write(f"**{comment['author']}** ({comment['score']} points)")
                        st.write(comment["body"])
                        render_replies(comment["replies"])
                        st.write("---")

            if grouped_comments["NEUTRAL"]:
                with st.expander("🟨 Neutral Arguments", expanded=False):
                    st.markdown(stance_summaries["NEUTRAL"])
                    st.write("---")
                    for comment in grouped_comments["NEUTRAL"]:
                        st.write(f"**{comment['author']}** ({comment['score']} points)")
                        st.write(comment["body"])
                        render_replies(comment["replies"])
                        st.write("---")
    
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    st.divider()
    
    # Step 4: Build knowledge graph
    if process_state["kg_info"] is None and process_state["grouped_comments"]:
        with st.spinner("Building Knowledge Graph..."):
            thread_data["classified_comments"] = process_state["grouped_comments"]

            kg_response = requests.post(
                kgCreator_endpoint,
                json={"thread_data": thread_data}
            )

            kg_info = {
                "success": False,
                "discussion_id": "",
                "node_counts": {},
                "action": ""
            }
            
            if kg_response.status_code == 200:
                kg_result = kg_response.json()
                if kg_result.get("status") == "success":
                    kg_info["success"] = True
                    kg_info["discussion_id"] = kg_result.get("discussion_id", "N/A")
                    kg_info["node_counts"] = kg_result.get("nodes_created", {})
                    kg_info["action"] = kg_result.get("action", "created")
                else:
                    st.error(f"Error building knowledge graph: {kg_result.get('message', 'Unknown error')}")
            else:
                st.error(f"Failed to build knowledge graph: {kg_response.text}")
                
            process_state["kg_info"] = kg_info
    
    # Display knowledge graph info as soon as we have it
    if process_state["kg_info"] and process_state["kg_info"]["success"]:
        kg_info = process_state["kg_info"]
        
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        st.subheader("Knowledge Graph")
        st.write(f"🧠 **Nodes Added:** Comments: {kg_info['node_counts'].get('comments', 0)}, "
                f"Replies: {kg_info['node_counts'].get('replies', 0)}, "
                f"Arguments: {kg_info['node_counts'].get('arguments', 0)}")

        if kg_info["action"] == "created":
            st.success("✅ Knowledge graph created from this discussion!")
        elif kg_info["action"] == "updated":
            st.success("♻️ Knowledge graph updated with comments/replies.")
    
    # When all processing is complete, save the discussion
    if (process_state["topic"] and 
        process_state["thread_data"] and 
        process_state["grouped_comments"] and 
        process_state["stance_summaries"] and 
        process_state["stance_percentages"] and 
        process_state["kg_info"]):
        
        # Store the complete discussion data
        complete_discussion = {
            "url": reddit_url,
            "topic": process_state["topic"],
            "thread_data": process_state["thread_data"],
            "grouped_comments": process_state["grouped_comments"],
            "stance_summaries": process_state["stance_summaries"],
            "stance_percentages": process_state["stance_percentages"],
            "kg_info": process_state["kg_info"]
        }
        
        # Add to discussions if not already there
        if all(d.get("url") != reddit_url for d in st.session_state.discussions):
            st.session_state.discussions.append(complete_discussion)
        else:
            # Update existing discussion
            for i, disc in enumerate(st.session_state.discussions):
                if disc.get("url") == reddit_url:
                    st.session_state.discussions[i] = complete_discussion
                    break
        

# Discussion View Page - for viewing saved discussions
elif st.session_state.page == 'discussion_view':
    if st.session_state.selected_discussion:
        discussion = st.session_state.selected_discussion
        
        # Extract all the data we need from the selected discussion
        topicIdentifier = discussion["topic"]
        thread_data = discussion["thread_data"]
        grouped_comments = discussion["grouped_comments"]
        stance_summaries = discussion["stance_summaries"]
        stance_percentages = discussion["stance_percentages"]
        kg_info = discussion.get("kg_info", {"success": False})
        
        # Display the header and basic info
        st.header(topicIdentifier)
        st.write(f"**Subreddit:** r/{thread_data['post']['subreddit']}")
        st.write(f"**Number of Upvotes:** {thread_data['post']['score']}")
        st.write(f"**Number of Comments:** {thread_data['post']['num_comments']}")

        # Original post content
        with st.expander("View Original Post Content", expanded=False):
            st.write(thread_data['post']['selftext'])
            
        # Stance distribution visualization
        bar_style = f"""
            <div style="display: flex; width: 100%; height: 25px; 
                        border-radius: 15px; overflow: hidden; 
                        border: 2px solid #ddd; margin-bottom: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">
                <div style="width: {stance_percentages['FOR']}%; background: linear-gradient(to right, #27ae60, #2ecc71);"></div>
                <div style="width: {stance_percentages['NEUTRAL']}%; background: linear-gradient(to right, #f1c40f, #f39c12);"></div>
                <div style="width: {stance_percentages['AGAINST']}%; background: linear-gradient(to right, #e74c3c, #c0392b);"></div>
            </div>
        """
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        st.divider()
        st.subheader("📊 Stance Distribution")
        st.markdown(bar_style, unsafe_allow_html=True)

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
        st.divider()

        # Display argument summaries
        st.subheader("Arguments by Stance - Summary")
        col1, col2 = st.columns([1.5, 1.5])

        with col1:
            st.markdown("### 🟩 Favorable Arguments")
            st.markdown(stance_summaries["FOR"])

        with col2:
            st.markdown("### 🟥 Against Arguments")
            st.markdown(stance_summaries["AGAINST"])

        expander_col1, expander_col2 = st.columns([1.5, 1.5])

        # Display original comments
        with expander_col1:
            with st.expander("Original Favorable Comments", expanded=False):
                for comment in grouped_comments["FOR"]:
                    st.write(f"**{comment['author']}** ({comment['score']} points)")
                    st.write(comment["body"])
                    render_replies(comment["replies"])
                    st.write("---")

        with expander_col2:
            with st.expander("Original Opposing Comments", expanded=False):
                for comment in grouped_comments["AGAINST"]:
                    st.write(f"**{comment['author']}** ({comment['score']} points)")
                    st.write(comment["body"])
                    render_replies(comment["replies"])
                    st.write("---")

        if grouped_comments["NEUTRAL"]:
            with st.expander("🟨 Neutral Arguments", expanded=False):
                st.markdown(stance_summaries["NEUTRAL"])
                st.write("---")
                for comment in grouped_comments["NEUTRAL"]:
                    st.write(f"**{comment['author']}** ({comment['score']} points)")
                    st.write(comment["body"])
                    render_replies(comment["replies"])
                    st.write("---")
                    
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        st.divider()

        # Display Knowledge Graph info if it was successful
        if kg_info.get("success", False):
            st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
            st.subheader("Knowledge Graph")
            st.write(f"🧠 **Nodes Added:** Comments: {kg_info['node_counts'].get('comments', 0)}, "
                    f"Replies: {kg_info['node_counts'].get('replies', 0)}, "
                    f"Arguments: {kg_info['node_counts'].get('arguments', 0)}")

            if kg_info.get("action") == "created":
                st.success("✅ Knowledge graph created from this discussion!")
            elif kg_info.get("action") == "updated":
                st.success("♻️ Knowledge graph updated with comments/replies.")
    else:
        st.error("No discussion selected. Please go back and select a discussion.")