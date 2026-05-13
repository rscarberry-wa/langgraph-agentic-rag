from dotenv import load_dotenv
import os

load_dotenv()

from graph.construct_graph import app

if __name__ == "__main__":
    # None of the documents are relevant in this case, so it'll have to perform a web search
    print(app.invoke(input={"question": "What is the economic outlook for France in 2026?"}))
    # All the docs will be relevant, so it will not do a web search
    print(app.invoke(input={"question": "What is a prominent AI trend?"}))