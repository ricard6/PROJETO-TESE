from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# SerpAPI Configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# FastAPI router
router = APIRouter()

# Request model
class ArticleRequest(BaseModel):
    topic: str

# Response model
class Article(BaseModel):
    title: str
    link: str
    snippet: str

class ArticleResponse(BaseModel):
    articles: List[Article]

@router.post("/fetchArticles", response_model=ArticleResponse)
def fetch_articles(request: ArticleRequest):
    params = {
        "q": request.topic,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "hl": "en"
    }

    try:
        response = requests.get(SERPAPI_ENDPOINT, params=params)
        data = response.json()

        if "organic_results" not in data:
            raise HTTPException(status_code=404, detail="No articles found.")

        articles = []
        for result in data["organic_results"][:5]:
            articles.append(Article(
                title=result.get("title", "No Title"),
                link=result.get("link", "No Link"),
                snippet=result.get("snippet", "No Description")
            ))

        return {"articles": articles}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
