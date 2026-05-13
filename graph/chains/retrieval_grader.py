import os

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model=f"ollama:{os.getenv('OLLAMA_MODEL')}",
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0.0,
)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'.")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_message = """
You are a grader assessing the relevance of a retrieved document to a question.
If the document contains keyword(s) or has a semantic meaning related to the question, grade it as relevant ('yes').
Otherwise, grade it as irrelevant ('no'). Your response should always be a binary score, 'yes' or 'no'.
"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "User question:\n\n {question}\n\nRetrieved document: \n\n{document}\n")
])

retrieval_grader = grade_prompt | structured_llm_grader
