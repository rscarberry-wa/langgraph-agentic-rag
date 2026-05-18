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

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check on generated answer."""
    binary_score: bool = Field(description="Generated text is grounded in the facts, not hallucinated, True or False.")

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader that assesses whether an LLM generated answer is grounded in/supported by the facts.
Provide a binary score of True or False, where True means the answer is grounded in the facts and False means it is hallucinated."""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts:\n\n {documents}\n\nLLM generation: \n\n{generation}\n")
])

hallucination_grader = hallucination_prompt | structured_llm_grader