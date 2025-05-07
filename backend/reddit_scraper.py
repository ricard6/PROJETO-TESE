from typing import Any, Dict
import requests                     # For making HTTP requests to external APIs (Reddit)
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel      # Base class for defining request/response schemas in FastAPI
import re

# Reddit Scraper Models
# Define the request model for scraping a Reddit thread
class RedditRequest(BaseModel):
    url: str

# Define the response model returned after scraping
class RedditResponse(BaseModel):
    thread_data: Dict[str, Any]
    

# Function to extract thread data
def fetch_reddit_data(url: str):
    """Fetch Reddit thread data and comments, keeping top 10 comments and their replies."""
    try:
        # Ensure the URL ends with a slash before appending '.json' for API access
        if not url.endswith('/'):
            url += '/'
        api_url = f"{url}.json"
        
        # Standard user-agent header to avoid being blocked by Reddit's servers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Send GET request to Reddit's API
        response = requests.get(api_url, headers=headers)
        
        # Raise HTTP error if request fails
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to access Reddit API: Status code {response.status_code}")
        
        # Parse JSON response (convert JSON response into python object) 
        data = response.json()
        
        # Extract the main post data
        post_data = data[0]['data']['children'][0]['data']
        # Extract the top-level comments
        comments_data = data[1]['data']['children']

        # Filter out non-comment entries and AutoModerator
        # Sort comments by score and take top 10
        sorted_comments = sorted(
            [c for c in comments_data if c.get('kind') == 't1' and c['data'].get('author') != "AutoModerator"], 
            key=lambda x: x['data'].get('score', 0), 
            reverse=True
        )[:10]  # Keep top 10 by score

        # Process comments and their replies (delegated to a helper function)
        processed_comments = process_comments(sorted_comments)

        # Build structured thread data
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
    
    # Catch any unexpected error and return a 500 error via FastAPI
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping Reddit thread: {str(e)}")
    

# Helper function to clean up post text
def process_text(text):
    """
    Process Reddit post text by marking quoted sections.

    - Adds '**Citing:**' before the first quoted line.
    - Adds '**End of Citation**' after the last quoted line.
    - Removes the quote symbol ('&gt;') from quoted lines.
    """
    lines = text.split("\n")    # Split the text into individual lines
    inside_citation = False     # Flag to track if we're inside a quoted block
    processed_lines = []

    for line in lines:
        # Detect if the line starts with a Reddit quote ('&gt;')
        if re.match(r"^\s*&gt;", line):  # If the line starts with '>'
            if not inside_citation:
                processed_lines.append("**Citing:**\n")  # Add citation start marker
                inside_citation = True
            # Remove '&gt;' and any extra spaces from the start of the line
            processed_lines.append(re.sub(r"^\s*&gt;\s*", "", line)) 
        else:
            if inside_citation:  
                processed_lines.append("\n**End of Citation**")  # Add citation end marker
                inside_citation = False
            processed_lines.append(line)

    # If the last lines were quotes, close the citation
    if inside_citation:  
        processed_lines.append("\n**End of Citation**")  

    return "\n".join(processed_lines)   # Reassemble processed text


# Helper function that processes comments 
def process_comments(comments_data, level=0, parent_body=None):
    """
    Recursively processes Reddit comments and their direct replies.

    Args:
        comments_data (list): List of comment dicts from Reddit JSON.
        level (int): Current nesting level (0 = top-level, increases with replies).
        parent_body (str): The body text of the parent comment, used for context.

    Returns:
        list: A list of processed comment dicts with metadata and nested replies.
    """
    processed = []

    # Skip anything that is not a comment 
    for comment in comments_data:
        if comment.get('kind') != 't1':  
            continue

        comment_data = comment['data']
        body_text = comment_data.get('body', '').strip()

        # Skip AutoModerator, system messages, deleted, or removed comments
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

        # If the comment has replies, process them recursively
        if comment_data.get('replies') and comment_data['replies'] != '':
            replies_data = comment_data['replies']['data']['children']
            comment_obj['replies'] = process_comments(replies_data, level + 1, formatted_body)

        processed.append(comment_obj)

    return processed



