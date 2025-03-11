import logging
from langchain.llms import Ollama
from crewai import Agent, Task
from app.retriever import Retriever
from app.config import LLM_MODEL

logging.basicConfig(level=logging.INFO)


class DSA:
    def __init__(self):
        """Initializes the DSA Agent with a dedicated FAISS index and CrewAI integration."""
        self.retriever = Retriever()
        self.faiss_index = "faiss_index/dsa"
        self.llm = Ollama(model=LLM_MODEL)

        # CrewAI Agent
        self.agent = Agent(
            name="DSA AI",
            role="Data Structures & Algorithms Expert",
            model=self.llm,
            description="Solves DSA problems, explains solutions, and optimizes code."
        )

    def query_rag(self, question):
        """Query the FAISS RAG model before task execution."""
        logging.info(f"DSA Agent retrieving relevant documents for: {question}")

        retrieved_docs = self.retriever.retrieve_for_agent("DSA", question)
        context = "\n".join(retrieved_docs)

        task_description = f"Use the following retrieved knowledge to solve this DSA problem:\n{context}\n\nProblem: {question}"
        task = Task(description=task_description, agent=self.agent)

        response = task.execute()
        is_sufficient = len(response.strip()) > 10 and "I don't know" not in response
        return response, is_sufficient


dsa_agent = DSA()
