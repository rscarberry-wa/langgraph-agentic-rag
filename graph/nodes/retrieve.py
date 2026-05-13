from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--- RETRIEVE ---")
    question = state["question"]
    docs = retriever.invoke(question)
    documents = [doc.page_content for doc in docs]
    return {"documents": documents, "question": question}