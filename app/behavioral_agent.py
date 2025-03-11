import logging
from langchain.llms import Ollama
from crewai import Agent, Task
from app.retriever import Retriever
from app.config import LLM_MODEL

logging.basicConfig(level=logging.INFO)


class Behavioral:
    def __init__(self):
        """Initializes the Behavioral Interview Agent with FAISS retrieval and CrewAI integration."""
        self.retriever = Retriever()
        self.faiss_index = "faiss_index/behavioral"
        self.llm = Ollama(model=LLM_MODEL)

        # CrewAI Agent
        self.agent = Agent(
            name="Behavioral AI",
            role="Behavioral Interview Coach",
            model=self.llm,
            description="Guides candidates through behavioral interviews using the STAR framework."
        )

    def coach_response(self, query):
        """Retrieve behavioral interview techniques before coaching the user."""
        logging.info(f"Behavioral Agent retrieving relevant documents for: {query}")

        retrieved_docs = self.retriever.retrieve_for_agent("Behavioral", query)
        context = "\n".join(retrieved_docs)

        task_description = f"Analyze the following retrieved STAR framework responses and coach the candidate accordingly:\n{context}\n\nCandidate Response: {query}"
        task = Task(description=task_description, agent=self.agent)

        return task.execute()


behavioral_agent = Behavioral()
