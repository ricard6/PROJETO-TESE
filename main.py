from fastapi import FastAPI
from backend.reddit_scraper import fetch_reddit_data, RedditRequest, RedditResponse
from backend.topic_identifier import topicIdentifier, TopicIdentifierRequest
from backend.summarize import SummarizeRequest, summarize_grouped_comments
from backend.stance_classification import stance_classifier, StanceClassificationRequest
from backend.kg_creator import KGRequest, create_knowledge_graph
from typing import Dict, List


# Define the API app
app = FastAPI()


# API Setup
# Summarization endpoint
@app.post("/summarizer")
async def summarize_comments_endpoint(request: Dict[str, Dict[str, List[str]]]):
    grouped_comments = request.get("grouped_comments", {})
    summaries = summarize_grouped_comments(grouped_comments)
    return {"summaries": summaries} 

# Reddit scraper endpoint
@app.post("/reddit_scraper", response_model=RedditResponse)
async def scrape_reddit_thread(request: RedditRequest):
    return fetch_reddit_data(request.url)

# Topic identifier endpoint
@app.post("/topicIdentifier")
async def identify_topic(request: TopicIdentifierRequest):
    return topicIdentifier(request)

# Stance classifier endpoint
@app.post("/stanceClassifier")
async def classify_stance(request: StanceClassificationRequest):
    return stance_classifier(request)

# Knowledge graph creator endpoint
@app.post("/kgCreator")
def build_kg(request: KGRequest):
    return create_knowledge_graph(request.thread_data)


