from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, List
import os
from pydantic import BaseModel

# Retrieve API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define request model
class SummarizeRequest(BaseModel):
    comments: List[str]

# Initialize summarizer model
summarizer = ChatOpenAI(
    model="gpt-4o",  
    openai_api_key=openai_api_key, 
    temperature=0.2
)

# Define summarization prompt (back to the original approach)
summarizer_prompt = PromptTemplate(
    input_variables=["comments"],
    template="""
    You are an expert at synthesizing online discussions. 
    Given the following user comments that share a similar stance in an online discussion, summarize the key points into a few concise bullet points.
    
    ### Guidelines:
    - Begin each bullet point with a short **title**, followed by a colon and the full idea.
    - The summary should be **self-contained** and reflect the main reasoning shared across the comments.
    - Do not repeat similar points. Focus on **distinct insights**.

    ### Example Output:

    - Economic Barriers: Many people want children but face high costs of childcare, healthcare, and housing.
    - Cultural Priorities: Modern values emphasize personal freedom and career growth, delaying or reducing interest in parenthood.
    - Policy Skepticism: Financial incentives like $5,000 are seen as insufficient or ineffective in changing birth rates.

    ### Comments:
    {comments}

    ### Summary (bullet points):
    """
)

# Create summarization chain
summary_chain = summarizer_prompt | summarizer

def summarize_grouped_comments(grouped_comments: Dict[str, List[str]]) -> Dict:
    """Summarizes FOR, AGAINST, and NEUTRAL comments separately."""
    summaries = {}

    for stance, comments in grouped_comments.items():
        if comments:
            summary_input = {"comments": "\n".join(comments)}
            summaries[stance] = summary_chain.invoke(summary_input).content.strip()
        else:
            summaries[stance] = "No significant arguments found."

    return summaries