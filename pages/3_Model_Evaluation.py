import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection
@st.cache_resource
def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        return driver
    except Exception as e:
        st.error(f"Error connecting to Neo4j: {e}")
        return None

# Initialize evaluator LLM
@st.cache_resource
def get_evaluator_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# Database query functions
def get_available_topics(driver):
    """Get all available discussion topics from Neo4j"""
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Topic)
            RETURN t.title AS title, t.discussion_id AS discussion_id, t.url AS url
            ORDER BY t.title
        """)
        return [(record["title"], record["discussion_id"], record["url"]) for record in result]

def get_discussion_data(driver, discussion_id: str):
    """Get complete discussion data for evaluation including replies"""
    with driver.session() as session:
        # Get topic info
        topic_result = session.run("""
            MATCH (t:Topic {discussion_id: $discussion_id})
            RETURN t.title AS title, t.url AS url
        """, discussion_id=discussion_id)
        topic = topic_result.single()
        
        # Get comments and their extracted arguments
        comments_result = session.run("""
            MATCH (c:Comment {discussion_id: $discussion_id})
            OPTIONAL MATCH (a:Argument {discussion_id: $discussion_id})-[:EXTRACTED_FROM]->(c)
            RETURN c.id AS comment_id, c.body AS comment_body, c.author AS author, 
                   c.score AS score, collect(a.text) AS arguments
            ORDER BY c.score DESC
        """, discussion_id=discussion_id)
        
        comments = []
        for record in comments_result:
            comments.append({
                "id": record["comment_id"],
                "body": record["comment_body"],
                "author": record["author"],
                "score": record["score"],
                "arguments": [arg for arg in record["arguments"] if arg],
                "type": "comment"
            })
        
        # Get replies and their extracted arguments
        replies_result = session.run("""
            MATCH (r:Reply {discussion_id: $discussion_id})
            OPTIONAL MATCH (a:Argument {discussion_id: $discussion_id})-[:EXTRACTED_FROM]->(r)
            OPTIONAL MATCH (r)-[:REPLY_TO]->(c:Comment)
            RETURN r.id AS reply_id, r.body AS reply_body, r.author AS author, 
                   r.score AS score, r.parent_comment_id AS parent_comment_id,
                   c.id AS parent_comment_actual_id, collect(a.text) AS arguments
            ORDER BY r.score DESC
        """, discussion_id=discussion_id)
        
        replies = []
        for record in replies_result:
            replies.append({
                "id": record["reply_id"],
                "body": record["reply_body"],
                "author": record["author"],
                "score": record["score"],
                "parent_comment_id": record["parent_comment_id"],
                "parent_comment_actual_id": record["parent_comment_actual_id"],
                "arguments": [arg for arg in record["arguments"] if arg],
                "type": "reply"
            })
        
        # Get all arguments and their clusters
        clusters_result = session.run("""
            MATCH (a:Argument {discussion_id: $discussion_id})-[:HAS_GROUP]->(g:ArgumentGroup)
            RETURN g.summary AS cluster_summary, g.stance AS stance, 
                   collect(a.text) AS arguments, count(a) AS argument_count
            ORDER BY g.stance, argument_count DESC
        """, discussion_id=discussion_id)
        
        clusters = []
        for record in clusters_result:
            clusters.append({
                "summary": record["cluster_summary"],
                "stance": record["stance"],
                "arguments": record["arguments"],
                "count": record["argument_count"]
            })
        
        # Get individual arguments with source info
        arguments_result = session.run("""
            MATCH (a:Argument {discussion_id: $discussion_id})
            OPTIONAL MATCH (a)-[:EXTRACTED_FROM]->(source)
            OPTIONAL MATCH (a)-[:HAS_GROUP]->(g:ArgumentGroup)
            RETURN a.text AS text, a.stance AS stance, 
                   labels(source)[0] AS source_type, source.id AS source_id,
                   g.summary AS cluster_summary
            ORDER BY a.stance, a.text
        """, discussion_id=discussion_id)
        
        arguments = []
        for record in arguments_result:
            arguments.append({
                "text": record["text"],
                "stance": record["stance"],
                "source_type": record["source_type"],
                "source_id": record["source_id"],
                "cluster_summary": record["cluster_summary"]
            })
        
        return {
            "topic": topic,
            "comments": comments,
            "replies": replies,
            "clusters": clusters,
            "arguments": arguments
        }
    
# Prompt for stance classification evaluation
STANCE_EVALUATION_PROMPT = PromptTemplate(
    input_variables=["topic", "argument_text", "detected_stance"],
    template="""
    You are evaluating the accuracy of stance classification for an argument used in the context of the following topic:
    "{topic}"

    Extracted argument:
    "{argument_text}"

    Stance assigned by the system:
    {detected_stance}

    Evaluate the correctness of the classification based on the argument content and the topic in question. Consider:

    - FOR: the argument supports or agrees with the main statement of the topic.
    - AGAINST: the argument disagrees with or contradicts the main statement of the topic.
    - NEUTRAL: the argument is informative, tangential, or does not take a clear position.

    Evaluation scale:
    1 = Incorrect (completely wrong classification)
    2 = Inaccurate (indicates wrong direction or is ambiguous)
    3 = Acceptable (can be interpreted as correct, but with doubts)
    4 = Correct (classification appropriately matches the content)
    5 = Very accurate (clearly correct and unambiguous classification)

    Respond in the format:
    Score: X
    Justification: [brief explanation in 1-2 sentences]
    """
)

# Evaluation prompts
ARGUMENT_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["content_text", "extracted_arguments", "content_type"],
    template="""
    You are evaluating the quality of argument extraction from a {content_type}.
    
    Original {content_type}:
    "{content_text}"
    
    Arguments Extracted by the System:
    {extracted_arguments}
    
    Evaluate the extraction based on:
    1. COMPLETENESS: Did the system capture all main arguments present in the text?
    2. ACCURACY: Are the extracted arguments actually present in the original text?
    3. QUALITY: Are the arguments well-formulated and self-contained?
    4. RELEVANCE: Are the extracted arguments actually arguments (not just vague statements)?
    
    Evaluation scale:
    1 = Very poor (missed important arguments or extracted irrelevant content)
    2 = Poor (captured some arguments but with many problems)
    3 = Average (captured main arguments but with some flaws)
    4 = Good (good extraction with minor problems)
    5 = Excellent (complete and accurate extraction)
    
    Respond in the format:
    Score: X
    Justification: [detailed explanation in 2-3 sentences]
    Identified problems: [list main problems, if any]
    """
)

CLUSTERING_EVALUATION_PROMPT = PromptTemplate(
    input_variables=["cluster_summary", "cluster_arguments"],
    template="""
    You are evaluating the quality of an argument cluster.
    
    Cluster Summary:
    "{cluster_summary}"
    
    Arguments in Cluster:
    {cluster_arguments}
    
    Evaluate the clustering based on:
    1. SEMANTIC COHERENCE: Do the grouped arguments really share the same central idea?
    2. SUMMARY QUALITY: Does the summary adequately capture the essence of the grouped arguments?
    3. GRANULARITY: Is the cluster neither too specific nor too generic?
    4. COMPLETENESS: Does the cluster include all relevant arguments for this idea?
    
    Evaluation scale:
    1 = Very poor (unrelated arguments grouped together)
    2 = Poor (some related arguments but confusing clustering)
    3 = Average (reasonable clustering but with some inconsistencies)
    4 = Good (coherent clustering with minor problems)
    5 = Excellent (semantically perfect clustering)
    
    Respond in the format:
    Score: X
    Justification: [detailed explanation in 2-3 sentences]
    Suggestions: [how to improve clustering, if applicable]
    """
)

def evaluate_stance(evaluator_llm, topic: str, argument_text: str, stance: str) -> Dict:
    prompt_text = STANCE_EVALUATION_PROMPT.format(
        topic=topic,
        argument_text=argument_text,
        detected_stance=stance
    )

    # DEBUG
    print("Prompt for stance evaluation:\n", prompt_text)

    response = evaluator_llm.invoke(prompt_text)

    # DEBUG
    print("LLM response:\n", response.content)

    content = response.content.strip()
    try:
        lines = content.split('\n')
        score_line = next(line for line in lines if line.startswith('Score:'))
        score = int(score_line.split(':')[1].strip())

        justification_line = next(line for line in lines if line.startswith('Justification:'))
        justification = justification_line.split(':', 1)[1].strip()

        return {
            "score": score,
            "justification": justification
        }
    except Exception as e:
        print(f"Error processing evaluation: {e}")
        print("Received content:", content)
        return {
            "score": 0,
            "justification": "Error processing evaluation"
        }

def evaluate_argument_extraction(evaluator_llm, content_body: str, extracted_arguments: List[str], content_type: str = "comment") -> Dict:
    """Evaluate argument extraction quality for a specific comment or reply"""
    if not extracted_arguments:
        return {
            "score": 1,
            "justification": f"No arguments were extracted from this {content_type}.",
            "problems": ["Empty extraction"]
        }
    
    args_text = "\n".join([f"- {arg}" for arg in extracted_arguments])
    
    response = evaluator_llm.invoke(
        ARGUMENT_EXTRACTION_PROMPT.format(
            content_text=content_body,
            extracted_arguments=args_text,
            content_type=content_type
        )
    )
    
    # Parse response
    content = response.content.strip()
    try:
        lines = content.split('\n')
        score_line = [line for line in lines if line.startswith('Score:')][0]
        score = int(score_line.split(':')[1].strip())
        
        justification_line = [line for line in lines if line.startswith('Justification:')][0]
        justification = justification_line.split(':', 1)[1].strip()
        
        problems_line = [line for line in lines if line.startswith('Identified problems:')]
        problems = problems_line[0].split(':', 1)[1].strip() if problems_line else "No specific problems identified"
        
        return {
            "score": score,
            "justification": justification,
            "problems": problems
        }
    except:
        return {
            "score": 0,
            "justification": "Error processing evaluation",
            "problems": ["Parsing error"]
        }

def evaluate_clustering(evaluator_llm, cluster_summary: str, cluster_arguments: List[str]) -> Dict:
    """Evaluate clustering quality for a specific cluster"""
    args_text = "\n".join([f"- {arg}" for arg in cluster_arguments])
    
    response = evaluator_llm.invoke(
        CLUSTERING_EVALUATION_PROMPT.format(
            cluster_summary=cluster_summary,
            cluster_arguments=args_text
        )
    )
    
    # Parse response
    content = response.content.strip()
    try:
        lines = content.split('\n')
        score_line = [line for line in lines if line.startswith('Score:')][0]
        score = int(score_line.split(':')[1].strip())
        
        justification_line = [line for line in lines if line.startswith('Justification:')][0]
        justification = justification_line.split(':', 1)[1].strip()
        
        suggestions_line = [line for line in lines if line.startswith('Suggestions:')]
        suggestions = suggestions_line[0].split(':', 1)[1].strip() if suggestions_line else "No specific suggestions"
        
        return {
            "score": score,
            "justification": justification,
            "suggestions": suggestions
        }
    except:
        return {
            "score": 0,
            "justification": "Error processing evaluation",
            "suggestions": "N/A"
        }

def main():
    st.set_page_config(page_title="Performance Evaluator - Neo4j", layout="wide")

    st.title("🔍 LLM Performance Evaluator")
    st.markdown("**Analysis based on real data stored in Neo4j (Comments + Replies)**")

    driver = get_neo4j_driver()
    if not driver:
        st.stop()

    evaluator_llm = get_evaluator_llm()
    topics = get_available_topics(driver)

    if not topics:
        st.warning("No topics found in the database.")
        st.stop()

    st.subheader("📋 Topic Selection")
    topic_options = [f"{title}" for title, discussion_id, url in topics]
    selected_topic_idx = st.selectbox("Choose the topic to evaluate:", range(len(topic_options)), format_func=lambda x: topic_options[x])

    selected_topic_title, selected_discussion_id, selected_url = topics[selected_topic_idx]

    with st.spinner("Loading discussion data..."):
        discussion_data = get_discussion_data(driver, selected_discussion_id)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Argument Extraction", "🎯 Clustering", "🧭 Stance Classification"])

    with tab1:
        st.subheader("📈 General Statistics")

        # General stance classification evaluation
        if st.button("📡 Evaluate Stance Classification (General)"):
            with st.spinner("Evaluating stance classifications..."):
                stance_scores = []
                for arg in discussion_data["arguments"]:
                    if arg.get("source_type") in ["Comment", "Reply"]:
                        evaluation = evaluate_stance(
                            evaluator_llm,
                            selected_topic_title,  
                            arg["text"],
                            arg["stance"]
                        )
                        stance_scores.append(evaluation["score"])

                if stance_scores:
                    avg_stance = sum(stance_scores) / len(stance_scores)
                    st.metric("🧭 Stance Classification", f"{avg_stance:.2f}/5")
                    st.progress(avg_stance / 5)
                else:
                    st.warning("No arguments with comment or reply source found for stance evaluation.")
    
    with tab1:
        st.subheader("📈 General Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Comments", len(discussion_data["comments"]))
        with col2:
            st.metric("Replies", len(discussion_data["replies"]))
        with col3:
            st.metric("Total Arguments", len(discussion_data["arguments"]))
        with col4:
            st.metric("Clusters", len(discussion_data["clusters"]))
        with col5:
            # Calculate average arguments per content (comments + replies)
            all_content = discussion_data["comments"] + discussion_data["replies"]
            total_args = sum(len(c["arguments"]) for c in all_content)
            avg_args_per_content = total_args / len(all_content) if all_content else 0
            st.metric("Args/Comments & Replies", f"{avg_args_per_content:.1f}")
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            comments_with_args = len([c for c in discussion_data["comments"] if c["arguments"]])
            st.metric("Comments w/ Args", f"{comments_with_args}/{len(discussion_data['comments'])}")
        with col2:
            replies_with_args = len([r for r in discussion_data["replies"] if r["arguments"]])
            st.metric("Replies w/ Args", f"{replies_with_args}/{len(discussion_data['replies'])}")
        with col3:
            total_content = len(discussion_data["comments"]) + len(discussion_data["replies"])
            st.metric("Total Content", total_content)
        
        # Distribution by stance
        if discussion_data["arguments"]:
            stance_counts = {}
            for arg in discussion_data["arguments"]:
                stance = arg["stance"]
                stance_counts[stance] = stance_counts.get(stance, 0) + 1
            
            fig = px.pie(values=list(stance_counts.values()), names=list(stance_counts.keys()), 
                        title="Argument Distribution by Stance")
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by source type
        if discussion_data["arguments"]:
            source_counts = {"Comment": 0, "Reply": 0}
            for arg in discussion_data["arguments"]:
                source_type = arg.get("source_type")
                if source_type == "Comment":
                    source_counts["Comment"] += 1
                elif source_type == "Reply":
                    source_counts["Reply"] += 1
            
            fig2 = px.bar(x=list(source_counts.keys()), y=list(source_counts.values()),
                         title="Argument Distribution by Source Type",
                         labels={"x": "Source Type", "y": "Number of Arguments"})
            st.plotly_chart(fig2, use_container_width=True)
        
        # Evaluate overall performance
        if st.button("🚀 Evaluate Overall Performance"):
            with st.spinner("Evaluating argument extraction..."):
                extraction_scores = []
                
                # Evaluate comments
                for comment in discussion_data["comments"]:
                    if comment["arguments"]:
                        eval_result = evaluate_argument_extraction(evaluator_llm, comment["body"], comment["arguments"], "comment")
                        extraction_scores.append(eval_result["score"])
                
                # Evaluate replies
                for reply in discussion_data["replies"]:
                    if reply["arguments"]:
                        eval_result = evaluate_argument_extraction(evaluator_llm, reply["body"], reply["arguments"], "reply")
                        extraction_scores.append(eval_result["score"])
                
                clustering_scores = []
                for cluster in discussion_data["clusters"]:
                    eval_result = evaluate_clustering(evaluator_llm, cluster["summary"], cluster["arguments"])
                    clustering_scores.append(eval_result["score"])

                stance_scores = []
                for arg in discussion_data["arguments"]:
                    if arg.get("source_type") in ["Comment", "Reply"]:
                        eval_result = evaluate_stance(
                            evaluator_llm,
                            selected_topic_title,  # or enriched topic, if you have it
                            arg["text"],           # extracted argument text
                            arg["stance"]
                        )
                        stance_scores.append(eval_result["score"])
                            
            col1, col2, col3= st.columns(3)
            with col1:
                avg_extraction = sum(extraction_scores) / len(extraction_scores) if extraction_scores else 0
                st.metric("📝 Argument Extraction", f"{avg_extraction:.2f}/5")
                st.progress(avg_extraction / 5)
            
            with col2:
                avg_clustering = sum(clustering_scores) / len(clustering_scores) if clustering_scores else 0
                st.metric("🎯 Clustering", f"{avg_clustering:.2f}/5")
                st.progress(avg_clustering / 5)
            
            with col3:
                avg_stance = sum(stance_scores) / len(stance_scores) if stance_scores else 0
                st.metric("🧭 Stance Classification", f"{avg_stance:.2f}/5")
                st.progress(avg_stance / 5) 
    
    with tab2:
        st.subheader("🔍 Argument Extraction Evaluation")
        
        # Combine comments and replies with arguments
        all_content_with_args = []
        
        # Add comments
        for i, comment in enumerate(discussion_data["comments"]):
            if comment["arguments"]:
                all_content_with_args.append({
                    "display": f"💬 Comment {i+1}: {comment['body'][:50]}...",
                    "content": comment,
                    "type": "comment",
                    "index": i
                })
        
        # Add replies
        for i, reply in enumerate(discussion_data["replies"]):
            if reply["arguments"]:
                parent_info = f" (reply to: {reply.get('parent_comment_id', 'unknown')})"
                all_content_with_args.append({
                    "display": f"↳ Reply {i+1}: {reply['body'][:50]}...{parent_info}",
                    "content": reply,
                    "type": "reply",
                    "index": i
                })
        
        if all_content_with_args:
            content_options = [item["display"] for item in all_content_with_args]
            selected_content_idx = st.selectbox("Select content:", range(len(content_options)), format_func=lambda x: content_options[x])
            
            selected_item = all_content_with_args[selected_content_idx]
            selected_content = selected_item["content"]
            content_type = selected_item["type"]
            
            st.markdown(f"**Original {content_type.title()}:**")
            st.text_area("", selected_content["body"], height=100, disabled=True)
            
            # Show parent context for replies
            if content_type == "reply" and selected_content.get("parent_comment_id"):
                st.markdown("**Context (Parent Comment):**")
                parent_comment = next((c for c in discussion_data["comments"] if c["id"] == selected_content.get("parent_comment_actual_id")), None)
                if parent_comment:
                    st.text_area("Parent comment:", parent_comment["body"][:200] + "...", height=100, disabled=True)
            
            st.markdown("**Extracted Arguments:**")
            for i, arg in enumerate(selected_content["arguments"]):
                st.write(f"{i+1}. {arg}")
            
            if st.button("Evaluate Extraction"):
                with st.spinner("Evaluating..."):
                    evaluation = evaluate_argument_extraction(evaluator_llm, selected_content["body"], selected_content["arguments"], content_type)

                
                score = evaluation["score"]
                score_colors = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🟢"}
                
                st.markdown(f"### {score_colors.get(score, '⚪')} Score: {score}/5")
                st.progress(score / 5)
                
                st.markdown("**Justification:**")
                st.info(evaluation["justification"])
                
                st.markdown("**Identified Problems:**")
                st.warning(evaluation["problems"])
        else:
            st.warning("No comments or replies with extracted arguments found.")
    
    with tab3:
        st.subheader("🎯 Clustering Evaluation")
        
        if discussion_data["clusters"]:
            cluster_options = [f"{c['stance']}: {c['summary'][:200]} ({c['count']} args)" for c in discussion_data["clusters"]]
            selected_cluster_idx = st.selectbox("Select a cluster:", range(len(cluster_options)), format_func=lambda x: cluster_options[x])
            
            selected_cluster = discussion_data["clusters"][selected_cluster_idx]
            
            st.markdown(f"**Stance:** {selected_cluster['stance']}")
            st.markdown("**Cluster Summary:**")
            st.text_area("", selected_cluster["summary"], height=80, disabled=True)
            
            st.markdown("**Arguments in Cluster:**")
            for i, arg in enumerate(selected_cluster["arguments"]):
                st.write(f"{i+1}. {arg}")
            
            # Show source distribution for this cluster
            cluster_args = selected_cluster["arguments"]
            comment_sources = 0
            reply_sources = 0
            
            for arg_text in cluster_args:
                matching_arg = next((a for a in discussion_data["arguments"] if a["text"] == arg_text), None)
                if matching_arg:
                    if matching_arg.get("source_type") == "Comment":
                        comment_sources += 1
                    elif matching_arg.get("source_type") == "Reply":
                        reply_sources += 1
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Args from Comments", comment_sources)
            with col2:
                st.metric("Args from Replies", reply_sources)
            
            if st.button("Evaluate Cluster"):
                with st.spinner("Evaluating..."):
                    evaluation = evaluate_clustering(evaluator_llm, selected_cluster["summary"], selected_cluster["arguments"])
                
                score = evaluation["score"]
                score_colors = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🟢"}
                
                st.markdown(f"### {score_colors.get(score, '⚪')} Score: {score}/5")
                st.progress(score / 5)
                
                st.markdown("**Justification:**")
                st.info(evaluation["justification"])
                
                st.markdown("**Suggestions:**")
                st.success(evaluation["suggestions"])
        else:
            st.warning("No clusters found for this topic.")

    with tab4:
        st.subheader("🧭 Stance Classification Evaluation")

        all_argument_sources = [arg for arg in discussion_data["arguments"] if arg.get("source_type") in ["Comment", "Reply"]]

        if all_argument_sources:
            selected_idx = st.selectbox(
                "Select an argument for evaluation:",
                range(len(all_argument_sources)),
                format_func=lambda i: all_argument_sources[i]["text"][:80] + "..."
            )

            selected_arg = all_argument_sources[selected_idx]

            st.markdown("**Text from Comment or Reply:**")
            matched_source = next(
                (item for item in (discussion_data["comments"] + discussion_data["replies"]) if item["id"] == selected_arg["source_id"]),
                None
            )
            if matched_source:
                st.text_area("", matched_source["body"], height=120, disabled=True)

            st.markdown("**Assigned stance:**")
            st.code(selected_arg["stance"])

            if st.button("Evaluate Stance"):
                with st.spinner("Evaluating stance classification..."):
                    evaluation = evaluate_stance(evaluator_llm, selected_topic_title, selected_arg["text"], selected_arg["stance"])

                score = evaluation["score"]
                st.markdown(f"### 🧭 Score: {score}/5")
                st.progress(score / 5)

                st.markdown("**Justification:**")
                st.info(evaluation["justification"])
        else:
            st.warning("No arguments with comment or reply source found.")

if __name__ == "__main__":
    main()