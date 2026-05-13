from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("--- GENERATE ---")
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join([doc for doc in documents])
    generation = generation_chain.invoke({"question": question, "context": context})
    return {"question": question, "documents": documents, "generation": generation}
