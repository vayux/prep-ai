import logging
from langchain.llms import Ollama
from crewai import Agent, Task
from app.retriever import Retriever
from app.config import LLM_MODEL

logging.basicConfig(level=logging.INFO)


class LLD:
    def __init__(self):
        """Initializes the LLD Agent with FAISS retrieval and CrewAI task execution."""
        self.retriever = Retriever()
        self.faiss_index = "faiss_index/lld"
        self.llm = Ollama(model=LLM_MODEL)

        # CrewAI Agent
        self.agent = Agent(
            name="LLD AI",
            role="Low-Level Design Expert",
            model=self.llm,
            description="Provides LLD diagrams, design patterns, and best practices."
        )

    def generate_lld(self, query):
        """Retrieve LLD-specific knowledge before generating an answer."""
        logging.info(f"LLD Agent retrieving relevant documents for: {query}")

        retrieved_docs = self.retriever.retrieve_for_agent("LLD", query)
        context = "\n".join(retrieved_docs)

        task_description = f"Use the retrieved LLD knowledge below to design a solution:\n{context}\n\nRequirement: {query}"
