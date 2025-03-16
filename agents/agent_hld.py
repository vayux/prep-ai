"""
HLD (High-Level Design) agent for PrepAI.
"""

import logging
from vector_store.data_retriever import DataRetriever


class HLDAgent:
    """Agent responsible for High-Level Design (HLD) queries."""

    def __init__(self, llm) -> None:
        """
        Args:
            llm: A callable LLM client for generating HLD responses.
        """
        self.llm = llm
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_retriever = DataRetriever()

    def propose_hld_architecture(self, user_input: str) -> str:
        """Retrieves context and returns a high-level architecture proposal.

        Args:
            user_input: HLD request, e.g. "Design a large-scale messaging platform."

        Returns:
            LLM-generated answer with relevant context from FAISS.
        """
        docs = self.data_retriever.similarity_search(user_input, k=5)
        context = "\n".join([d.page_content for d in docs if d and d.page_content])
        prompt = (
            f"Context:\n{context}\n\n"
            f"HLD Request: {user_input}\n"
            "Outline a robust, high-level system architecture."
        )
        return self.llm(prompt)
