from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

import os

class RouteQuery(BaseModel):
    """Route user question to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ..., # To make it required
        description="Given a user question, choose to route it to either vectorstore or websearch."
    )

llm = init_chat_model(
    model=f"ollama:{os.getenv('OLLAMA_MODEL')}",
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0.0,
)

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user's question to a vectorstore or a web search.
The vectorstore contains documents related to AI agents and AI trends. Use the vectorstore for
these and closely related topics. Use websearch for everything else."""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

question_router = router_prompt | structured_llm_router