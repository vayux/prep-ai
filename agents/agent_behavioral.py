"""
Behavioral agent for PrepAI, handling soft-skill and HR-style questions.
"""

import logging
from vector_store.data_retriever import DataRetriever


class BehavioralAgent:
    """Agent responsible for Behavioral (soft-skill) questions."""

    def __init__(self, llm) -> None:
        """
        Args:
            llm: A callable LLM client for generating responses.
        """
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_retriever = DataRetriever()

    def handle_behavioral_question(self, user_input: str) -> str:
        """Retrieves context and returns an appropriate behavioral answer.

        Args:
            user_input: The user question or scenario, e.g. "How do you handle conflict at work?"

        Returns:
            LLM-generated answer with relevant context from FAISS if available.
        """
        docs = self.data_retriever.similarity_search(user_input, k=2)
        context = "\n".join([d.page_content for d in docs if d and d.page_content])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Behavioral Question: {user_input}\n"
            "Answer politely and with real-world examples."
        )
        return self.llm(prompt)
