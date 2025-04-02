from typing import Any, Dict
import requests
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel
import re


# Reddit scraper models
class RedditRequest(BaseModel):
    url: str

class RedditResponse(BaseModel):
    thread_data: Dict[str, Any]
    

def fetch_reddit_data(url: str):
    """Fetch Reddit thread data and comments, keeping top 10 comments and their replies."""
    try:
        if not url.endswith('/'):
            url += '/'
        api_url = f"{url}.json"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(api_url, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to access Reddit API: Status code {response.status_code}")
        
        data = response.json()
        
        post_data = data[0]['data']['children'][0]['data']
        comments_data = data[1]['data']['children']

        # Sort comments by score and take top 10
        sorted_comments = sorted(
            [c for c in comments_data if c.get('kind') == 't1' and c['data'].get('author') != "AutoModerator"], 
            key=lambda x: x['data'].get('score', 0), 
            reverse=True
        )[:10]

        # Process only top 10 comments and their replies
        processed_comments = process_comments(sorted_comments)

        thread_data = {
            'post': {
                'title': post_data.get('title', ''),
                'author': post_data.get('author', ''),
                'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'score': post_data.get('score', 0),
                'upvote_ratio': post_data.get('upvote_ratio', 0),
                'url': post_data.get('url', ''),
                'selftext': process_text(post_data.get('selftext', '')),
                'num_comments': post_data.get('num_comments', 0),
                'subreddit': post_data.get('subreddit', '')
            },
            'comments': processed_comments
        }
        
        return {"thread_data": thread_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping Reddit thread: {str(e)}")
    

def process_text(text):
    """Add 'Citing:' on its own line before the first quote and 'End of Citation' after the last quote."""
    lines = text.split("\n")
    inside_citation = False
    processed_lines = []

    for line in lines:
        if re.match(r"^\s*&gt;", line):  # If the line starts with '>'
            if not inside_citation:
                processed_lines.append("**Citing:**\n")  # Ensure "Citing:" is on its own line
                inside_citation = True
            processed_lines.append(re.sub(r"^\s*&gt;\s*", "", line))  # Remove '>'
        else:
            if inside_citation:  
                processed_lines.append("\n**End of Citation**")  # Ensure "End of Citation" is on its own line
                inside_citation = False
            processed_lines.append(line)

    if inside_citation:  
        processed_lines.append("\n**End of Citation**")  # Ensure ending citation if last lines are quoted

    return "\n".join(processed_lines)


def process_comments(comments_data, level=0, parent_body=None):
    """Process comments recursively, keeping only the top 10 root comments by score and their direct replies."""
    processed = []

    for comment in comments_data:
        if comment.get('kind') != 't1':  # Ensure it's a comment
            continue

        comment_data = comment['data']
        body_text = comment_data.get('body', '').strip()

        # Skip AutoModerator, deleted, or removed comments
        if comment_data.get('author') == "AutoModerator" or body_text.lower() in ["[deleted]", "[removed]"]:
            continue

        # Apply citation formatting to comment body
        formatted_body = process_text(body_text)  

        # Build comment object
        comment_obj = {
            'id': comment_data.get('id', ''),
            'author': comment_data.get('author', ''),
            'created_utc': datetime.fromtimestamp(comment_data.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S') if comment_data.get('created_utc') else None,
            'body': formatted_body,
            'score': comment_data.get('score', 0),
            'level': level,
            'parent_body': parent_body,  # Include parent comment context
            'replies': []
        }

        # Process direct replies
        if comment_data.get('replies') and comment_data['replies'] != '':
            replies_data = comment_data['replies']['data']['children']
            comment_obj['replies'] = process_comments(replies_data, level + 1, formatted_body)

        processed.append(comment_obj)

    return processed



