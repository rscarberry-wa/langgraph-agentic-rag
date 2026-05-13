from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

llm = init_chat_model(
    model=f"ollama:{os.getenv('OLLAMA_MODEL')}",
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0.0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved content to answer the
        question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Question: {question}
        
        Context: {context}
        
        Answer:""")
    ]
)

generation_chain = prompt | llm | StrOutputParser()


