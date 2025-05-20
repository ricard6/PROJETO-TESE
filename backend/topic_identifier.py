from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

# Retrieve API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the request
class TopicIdentifierRequest(BaseModel):
    text:str    # The text to analyze

# Discussion topic identifier model (via Langchain)
topicIdentifier_model = ChatOpenAI(
    model = "gpt-4o-mini",
    max_tokens = 200,
    temperature = 0.1,
    openai_api_key = openai_api_key
)

# Function that identifies the discussion topic and stances 
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Text: "{text}"

    Instruction: This is a social media discussion thread header. 
    Based on it, identify the main topic being discussed in the thread, so that it is possible to know, reading the topic, what the two stances (for and against) would mean in that context.
    Then, really briefly explain what the "for" and "against" stances refer to in the context of this topic. 
    Specifically, really briefly describe what these stances represent regarding the actions or policies being discussed, and clarify what "for" and "against" are supporting or opposing."
    Response:
    """
)

# Combine prompt and model into a runnable chain
topic_chain = prompt | topicIdentifier_model

# Function that identifies the topic and describes both stances
def topicIdentifier(request: TopicIdentifierRequest):
    response = topic_chain.invoke({"text": request.text})
    topic_text = response.content.strip()  
    return {"topic": topic_text}