from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.answer_grader import GradeAnswer, answer_grader
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

def test_hallucination_grader_answer_yes() -> None:
    question = "What are some current Gen AI trends?"
    documents = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in documents])
    generation = generation_chain.invoke({"question": question, "context": context})
    response = hallucination_grader.invoke({"documents": context, "generation": generation})
    assert response.binary_score == True

def test_hallucination_grader_answer_no() -> None:
    question = "Why do ducks go quack?"
    documents = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in documents])
    generation = "Ducks go quack because they aren't storks."
    response = hallucination_grader.invoke({"documents": context, "generation": generation})
    assert response.binary_score == False

def test_answer_grader_answer_yes() -> None:
    question = "Given a circle of radius R, how do you find its area?"
    generation = "There area of a circle is pi * R^2."
    response = answer_grader.invoke({"question": question, "generation": generation})
    assert response.binary_score == True

def test_answer_grader_answer_no() -> None:
    question = "Given a circle of radius R, how do you find its area?"
    generation = "My favorite day of the week is Sunday."
    response = answer_grader.invoke({"question": question, "generation": generation})
    assert response.binary_score == False