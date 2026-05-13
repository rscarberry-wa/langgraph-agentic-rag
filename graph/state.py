from typing import List, TypedDict

from langchain_core.outputs import generation


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: str - The current question.
        generation: str - LLM generation.
        web_search: bool - Whether to use web search.
        documents: List[str] - List of documents.
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]
