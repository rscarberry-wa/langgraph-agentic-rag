from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain
from pprint import pprint

def test_retrieval_grader_answer_yes() -> None:
    question = "What are some current Gen AI trends?"
    documents = retriever.invoke(question)
    response = retrieval_grader.invoke({"question": question, "document": documents[0].page_content})
    assert response.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "How do you know when your prunes are ripe?"
    documents = retriever.invoke(question)
    if len(documents) > 0:
        response = retrieval_grader.invoke({"question": question, "document": documents[0].page_content})
        assert response.binary_score == "no"

def test_generation_chain() -> None:
    question = "What are some current Gen AI trends?"
    documents = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in documents])
    generation = generation_chain.invoke({"question": question, "context": context})
    pprint(generation)
    assert generation is not None
