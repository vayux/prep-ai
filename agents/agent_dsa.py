"""
DSA (Data Structures & Algorithms) agent for PrepAI.
"""

import logging
from vector_store.data_retriever import DataRetriever


class DSAAgent:
    """Agent responsible for DSA-related queries."""

    def __init__(self, llm) -> None:
        """
        Args:
            llm: A callable LLM client (e.g., Ollama or OpenAI) that takes a prompt
                 string and returns a string response.
        """
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_retriever = DataRetriever()

    def solve_problem(self, user_input: str) -> str:
        """Retrieves context from FAISS and returns a solution for a DSA query.

        Args:
            user_input: The user query, such as "Explain how to reverse a linked list."

        Returns:
            The LLM-generated answer incorporating retrieval-augmented context.
        """
        docs = self.data_retriever.similarity_search(user_input, k=5)
        context = "\n".join([d.page_content for d in docs if d and d.page_content])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {user_input}\n"
            "Please provide a detailed, step-by-step solution."
        )
        return self.llm(prompt)
