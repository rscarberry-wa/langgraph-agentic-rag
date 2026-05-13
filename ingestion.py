from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

load_dotenv()

urls = [
    "https://blog.segmind.com/32-qwen-models-on-segmind-the-full-model-family-overview/",
    "https://blog.segmind.com/how-to-use-the-entire-segmind-model-library-through-an-ai-agent/",
    "https://blog.segmind.com/gen-ai-trends/"
]

# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]
#
# print(f"Loaded {len(docs_list)} documents from {len(urls)} URLs")
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
#
# doc_splits = text_splitter.split_documents(docs_list)
#
# vector_store = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=OllamaEmbeddings(
#         model=os.getenv("RAG_EMBEDDING_MODEL"),
#         base_url=os.getenv("OLLAMA_BASE_URL")
#     ),
#     persist_directory="./.chroma"
# )

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory=os.path.join(_BASE_DIR, ".chroma"),
    embedding_function=OllamaEmbeddings(
        model=os.getenv("RAG_EMBEDDING_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL")
    )
).as_retriever()



