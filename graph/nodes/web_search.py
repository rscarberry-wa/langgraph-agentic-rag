from dotenv import load_dotenv
from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("--- WEB SEARCH ---")
    question = state["question"]
    documents = state["documents"]
    tavily_response = web_search_tool.invoke({
        "query": question
    })
    joined_tavily_result = "\n".join([tr["content"] for tr in tavily_response["results"]])
    web_results = [joined_tavily_result]
    if documents is not None:
        documents.extend(web_results)
    else:
        documents = web_results

    return {
        "question": question,
        "documents": documents,
    }

if __name__ == "__main__":
    newState = web_search(state = {
        "question": "What national capital is the most icky?",
        "documents": None,
        "web_search": True,
        "generation": None
    })
    print(newState)