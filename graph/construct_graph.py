from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from graph.nodes import *
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state: GraphState) -> str:
    print("--- DECIDE TO GENERATE ---")
    if state["web_search"]:
        print("  -- DECISION: NOT ALL DOCUMENTS ARE RELEVANT, WEB SEARCHING")
        return WEB_SEARCH
    else:
        print("  -- DECISION: ALL DOCUMENTS ARE RELEVANT, GENERATING")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("--- CHECK HALLUCINATIONS ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score =  hallucination_grader.invoke({
        "documents": "\n\n".join(documents),
        "generation": generation
    })
    if hallucination_grade := score.binary_score:
        print("  --- DECISION: ANSWER IS GROUNDED IN DOCUMENTS ---")
        print("  --- CHECK IF ANSWER IS RELEVANT ---")
        score = answer_grader.invoke({
            "question": question, "generation": generation
        })
        if answer_grade := score.binary_score:
            print("    --- DECISION: ANSWER IS RELEVANT ---")
            return "useful"
        else:
            print("    --- DECISION: ANSWER IS NOT RELEVANT, BACK TO WEB SEARCH ---")
            return "not useful"
    else:
        return "not supported"

def route_question(state: GraphState) -> str:
    print("--- ROUTE QUESTION ---")
    question = state["question"]
    route_query: RouteQuery = question_router.invoke({"question": question})
    if route_query.datasource == "vectorstore":
        print("  --- DECISION: ROUTE TO VECTORSTORE ---")
        return RETRIEVE
    elif route_query.datasource == "websearch":
        print("  --- DECISION: ROUTE TO WEBSEARCH ---")
        return WEB_SEARCH

workflow = StateGraph(state_schema=GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_conditional_entry_point(route_question, path_map={
    RETRIEVE: RETRIEVE,
    WEB_SEARCH: WEB_SEARCH,
})
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, path_map={WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE})
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question,
                               path_map={"useful": END, "not useful": WEB_SEARCH, "not supported": GENERATE})

app = workflow.compile()

# app.get_graph().draw_mermaid_png(output_file_path="graph.png")


