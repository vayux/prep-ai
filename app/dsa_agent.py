import logging
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from app.retriever import Retriever
from app.config import LLM_MODEL

logging.basicConfig(level=logging.INFO)


class DSA:
    def __init__(self):
        self.retriever = Retriever()
        self.qa_chain = None
        self.initialize_qa()

    def initialize_qa(self):
        """Initialize Retrieval-Augmented Generation (RAG)."""
        retriever = self.retriever.load_faiss_index()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=Ollama(model=LLM_MODEL), chain_type="stuff", retriever=retriever
        )

    def query_rag(self, question):
        """Query the RAG model and determine sufficiency."""
        response = self.qa_chain.run(question)

        # Implement your sufficiency logic here.
        is_sufficient = self.is_response_sufficient(response)

        return response, is_sufficient

    def is_response_sufficient(self, response):
        """Determines if the RAG response is sufficient."""
        # Example: Check if the response is too short or contains "I don't know".
        if len(response.strip()) < 10 or "I don't know" in response:
            return False
        else:
            return True


dsa_agent = DSA()
