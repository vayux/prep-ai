import logging
from langchain.llms import Ollama
from crewai import Agent, Task
from app.retriever import Retriever
from app.config import LLM_MODEL

logging.basicConfig(level=logging.INFO)


class HLD:
    def __init__(self):
        """Initializes the HLD Agent with FAISS retrieval and CrewAI task execution."""
        self.retriever = Retriever()
        self.faiss_index = "faiss_index/hld"
        self.llm = Ollama(model=LLM_MODEL)

        # CrewAI Agent
        self.agent = Agent(
            name="HLD AI",
            role="High-Level Design Expert",
            model=self.llm,
            description="Discusses architectures, scalability, and trade-offs in system design."
        )

    def generate_hld(self, query):
        """Retrieve HLD-specific knowledge before generating a response."""
        logging.info(f"HLD Agent retrieving relevant documents for: {query}")

        retrieved_docs = self.retriever.retrieve_for_agent("HLD", query)
        context = "\n".join(retrieved_docs)

        task_description = f"Use the retrieved
