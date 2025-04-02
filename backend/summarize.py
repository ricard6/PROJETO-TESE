from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
    model="gpt-4o-mini",  
    openai_api_key=openai_api_key, 
    temperature=0.2
)

# Define summarization prompt (back to the original approach)
summarizer_prompt = PromptTemplate(
    input_variables=["comments"],
    template="""
    You are an expert at synthesizing online discussions. 
    Given the following user comments that share a similar stance in an online discussion, summarize the key points into a few concise bullet points.

    **Comments:**
    {comments}

    **Summary (bullet points):**
    """
)

# Create summarization chain
summary_chain = LLMChain(
    llm=summarizer,
    prompt=summarizer_prompt
)

def summarize_grouped_comments(grouped_comments: Dict[str, List[str]]) -> Dict:
    """Summarizes FOR, AGAINST, and NEUTRAL comments separately."""
    
    summaries = {}

    for stance, comments in grouped_comments.items():
        if comments:
            summary_input = {"comments": "\n".join(comments)}
            summaries[stance] = summary_chain.run(summary_input)
        else:
            summaries[stance] = "No significant arguments found."

    return summaries