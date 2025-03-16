"""
LLD (Low-Level Design) agent for PrepAI.
"""

import logging
from vector_store.data_retriever import DataRetriever


class LLDAgent:
    """Agent responsible for Low-Level Design (LLD) queries."""

    def __init__(self, llm) -> None:
        """
        Args:
            llm: A callable LLM client for generating responses.
        """
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_retriever = DataRetriever()

    def propose_lld_solution(self, user_input: str) -> str:
        """Retrieves context from FAISS and returns a low-level design solution.

        Args:
            user_input: LLD request, e.g. "Design a rate limiter."

        Returns:
            LLM-generated answer with retrieval-augmented context.
        """
        docs = self.data_retriever.similarity_search(user_input, k=5)
        context = "\n".join([d.page_content for d in docs if d and d.page_content])
        prompt = (
            f"Context:\n{context}\n\n"
            f"LLD Request: {user_input}\n"
            "Provide a thorough low-level design."
        )
        return self.llm(prompt)
