from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
import os

llm = init_chat_model(
    model=f"ollama:{os.getenv('OLLAMA_MODEL')}",
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0.0,
)

class GradeAnswer(BaseModel):
    """Binary score whether an LLM generated response answers the user's question."""
    binary_score: bool = Field(description="Generated answer answers the user's question, True or False.")

structured_answer_grader = llm.with_structured_output(GradeAnswer)

system = """You are an answer grader that assesses whether an LLM generated answer answers the user's question.
Provide a binary score of True or False, where True means the answer answers the question, False if it does not."""

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "The user's question:\n\n {question}\n\nLLM generation: \n\n{generation}\n")
])

answer_grader = grader_prompt | structured_answer_grader