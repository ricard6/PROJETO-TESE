import streamlit as st
import requests

# App config
st.set_page_config(
        page_title="Discussion Analysis",
)

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
kgCreator_endpoint = f"{backend_url}/kgCreator"

# Button to process
if st.button("Go!", type="primary"):
    # Fazer o scrape da thread do Reddit
    with st.spinner("Initializing analysis..."):
        scrape_response = requests.post(
            reddit_scraper_endpoint,
            json={"url": reddit_url}
        )

    if scrape_response.status_code == 200:
        thread_data = scrape_response.json().get("thread_data")

        # Identificar o tópico da discussão passando os dados necessários ao LLM no "topic_identifier"
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

        # UI inicial com tópico identificado, nome do subreddit, número de upvotes e comentários da thread
        st.header(topicIdentifier)
        st.write(f"**Subreddit:** r/{thread_data['post']['subreddit']}")
        st.write(f"**Number of Upvotes:** {thread_data['post']['score']}")
        st.write(f"**Number of Comments:** {thread_data['post']['num_comments']}")

        # Expander com o post inicial da discussão  
        with st.expander("View Original Post Content", expanded=False):
            st.write(thread_data['post']['selftext'])

        # Criação dos arrays de comentarios por stance
        top_comments = thread_data['comments']

        grouped_comments = {"FOR": [], "AGAINST": [], "NEUTRAL": []}
        all_classified_bodies = {"FOR": [], "AGAINST": [], "NEUTRAL": []}

        # Loop para extração das top 5 replies aos comentários e classificação de stance
        for comment in top_comments:
            sorted_replies = sorted(comment['replies'], key=lambda x: x['score'], reverse=True)[:5]
            parent_context = f"Parent Comment: {comment.get('parent_body', 'N/A')}\n\n" if 'parent_body' in comment else ""

            # Classificação da stance dos comentários
            with st.spinner(f"Classifying comments..."):
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

                # Classificação da stance das replies aos comentários
                classified_replies = []
                for reply in sorted_replies:
                    with st.spinner(f"Classifying replies..."):
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

                    # Criação e armazenamento de um dictionary para cada reply com autor, score, body e stance
                    classified_replies.append({
                        "author": reply["author"],
                        "score": reply["score"],
                        "body": reply["body"],
                        "stance": reply_stance
                    })

                    # Junção do corpo da reply à stance para inclusão na sumarização
                    all_classified_bodies[reply_stance].append(reply["body"])
                
                # Organização de comments por stance
                # Criação e armazenamento de um dictionary para cada comment com autor, score, body e replies (já classificadas)
                grouped_comments[stance_result].append({
                    "author": comment["author"],
                    "score": comment["score"],
                    "body": comment["body"],
                    "replies": classified_replies
                })

                # Adição do texto de cada comment ao conjunto de textos da stance correspondente para sumarização por stance
                all_classified_bodies[stance_result].append(comment["body"])

        # Classificação dos comments e replies por stance
        with st.spinner("Analyzing arguments by stance..."):
            stance_summary_response = requests.post(
                summarizer_endpoint,
                json={"grouped_comments": all_classified_bodies}
            )

            stance_summaries = stance_summary_response.json().get("summaries", {}) if stance_summary_response.status_code == 200 else {
                "FOR": "Unable to summarize favorable arguments.",
                "AGAINST": "Unable to summarize opposing arguments.",
                "NEUTRAL": "Unable to summarize neutral arguments."
            }

        # Criação do meter de "Stance Distribution"
        stance_counts = {k: len(grouped_comments[k]) for k in grouped_comments}
        total_comments = sum(stance_counts.values())

        if total_comments > 0:
            stance_percentages = {
                k: (stance_counts[k] / total_comments) * 100 for k in stance_counts
            }
        else:
            stance_percentages = {k: 0 for k in grouped_comments}

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

        # Apresentação dos sumários dos argumentos separados por stance
        st.subheader("Arguments by Stance")
        col1, col2 = st.columns([1.5, 1.5])

        with col1:
            st.markdown("### 🟩 Favorable Arguments")
            st.markdown(stance_summaries["FOR"])

        with col2:
            st.markdown("### 🟥 Against Arguments")
            st.markdown(stance_summaries["AGAINST"])

        expander_col1, expander_col2 = st.columns([1.5, 1.5])

        # Apresentação das replies junto do comentário correspondente com indicador de stance
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

        # Expanders com os comentários originais de cada stance 
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

        # Debug the structure of the data being sent
        st.write(f"Found {len(grouped_comments.get('FOR', []))} FOR comments")
        st.write(f"Found {len(grouped_comments.get('AGAINST', []))} AGAINST comments")
        st.write(f"Found {len(grouped_comments.get('NEUTRAL', []))} NEUTRAL comments")

        with st.spinner("Building Knowledge Graph..."):
    
            thread_data["classified_comments"] = grouped_comments
            
            kg_response = requests.post(
                kgCreator_endpoint, 
                json={"thread_data": thread_data}
            )

            if kg_response.status_code == 200:
                kg_result = kg_response.json()
                if kg_result.get("status") == "success":
                    st.write(f"Number of nodes: {kg_result.get('nodes_created', {})}")
                    st.success("Knowledge graph successfully built from discussion!")
                else:
                    st.error(f"Error building knowledge graph: {kg_result.get('message', 'Unknown error')}")
            else:
                st.error(f"Failed to build knowledge graph: {kg_response.text}")
