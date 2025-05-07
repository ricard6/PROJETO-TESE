from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import os

# Retrieve API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define request model
class StanceClassificationRequest(BaseModel):
    thread_title: str
    thread_selftext: str
    identified_topic: str
    comment_body: str

# Initialize LangChain stance detection model
stance_model = ChatOpenAI(
    model="gpt-4o",  
    openai_api_key=openai_api_key,  
    temperature=0,
    max_tokens=10
)

# Define stance classification prompt using LangChain's PromptTemplate
stance_prompt = PromptTemplate(
    input_variables=["thread_title", "thread_selftext", "identified_topic", "comment_body"],
    template="""
    You are an AI trained in stance detection. Your task is to classify a Reddit comment's stance toward the discussion topic based on the full thread context.
    Analyze the given Reddit post and comment carefully, and return ONLY ONE of the following labels:
    - AGAINST
    - FOR
    - NEUTRAL

    Keep in mind that the comments can contain sarcasm, irony. Do NOT provide any explanation, analysis, or additional text.

    Reddit Thread:
    Title: "{thread_title}"
    Post Content: "{thread_selftext}"
    Identified Discussion Topic: "{identified_topic}"

    Reddit Comment:
    "{comment_body}"

    Label:
    """
)

# Create LangChain chain for stance classification
stance_chain = LLMChain(
    llm=stance_model,
    prompt=stance_prompt
)

# Function to classify stance
def stance_classifier(request: StanceClassificationRequest):
    response = stance_chain.run(
        thread_title=request.thread_title,
        thread_selftext=request.thread_selftext,
        identified_topic=request.identified_topic,
        comment_body=request.comment_body
    )
    
    return {"stance": response.strip()}