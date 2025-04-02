from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import os

# Retrieve API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

#Define the request
class TopicIdentifierRequest(BaseModel):
    text:str

# Discussion topic identifier
topicIdentifier_model = ChatOpenAI(
    model = "gpt-4o-mini",
    max_tokens = 200,
    temperature = 0.1,
    openai_api_key = openai_api_key
)


def topicIdentifier(request: TopicIdentifierRequest):
    # Prompt to identify the main topic in the discussion
    prompt = PromptTemplate(
        input_variables=["text"],
        template ="""
        Text: "{text}"

        Instruction: This is a social media discussion thread header. 
        Based on it, identify the main topic being discussed in the thread, so that it is possible to know, reading the topic, what the two stances (for and against) would mean in that context.
        Then, really briefly explain what the "for" and "against" stances refer to in the context of this topic. 
        Specifically, really briefly describe what these stances represent regarding the actions or policies being discussed, and clarify what "for" and "against" are supporting or opposing.".
        Response:
        """
    )

    # Directly use ChatOpenAI inside LLMChain
    topic_chain = LLMChain(
        llm = topicIdentifier_model,
        prompt = prompt
    )

    response = topic_chain.run(text=request.text)
    topic_text = response.strip()

    return {"topic": topic_text}