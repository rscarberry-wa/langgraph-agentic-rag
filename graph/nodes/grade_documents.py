from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines what the retrieved documents are relevant to the question.
    If any document is not relevant, we set a flag to run a web search.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        Dict[str, Any]: New state with irrelevant documents filtered out and
        web search flag set if necessary.
    """
    print("--- GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = False
    for doc in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": doc
        })
        grade = score.binary_score.lower()
        if grade == "no":
            print("--- GRADE: DOCUMENT NOT RELEVANT ---")
            web_search = True
        else:
            print("--- GRADE: DOCUMENT RELEVANT ---")
            filtered_docs.append(doc)

    return {
        "question": question,
        "documents": filtered_docs,
        "web_search": web_search or len(documents) == 0
    }