"""
FAISS-based data retrieval for PrepAI.
"""

import os
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import settings


class DataRetriever:
    """Manages FAISS indexing and retrieval for PrepAI."""

    def __init__(self, vector_store_path: str = None) -> None:
        """
        Args:
            vector_store_path: Optional custom path to an existing FAISS index.
        """
        self.vector_store_path = vector_store_path or settings.VECTOR_STORE_PATH
        self.embeddings_model = HuggingFaceEmbeddings()
        self.vector_store = self._load_or_create_vector_store()

    def _load_or_create_vector_store(self) -> FAISS:
        """Loads an existing FAISS index or creates a new one.

        Returns:
            A FAISS vector store object.
        """
        if os.path.exists(self.vector_store_path):
            return FAISS.load_local(self.vector_store_path, self.embeddings_model, allow_dangerous_deserialization=True)
        return FAISS.from_texts([""], self.embeddings_model)

    def add_documents(self, documents: list[str]) -> None:
        """Adds new documents to the vector store and saves it.

        Args:
            documents: A list of document strings to embed and store.
        """
        if not documents:
            return
        try:
            self.vector_store.add_texts(documents)
            self.vector_store.save_local(self.vector_store_path)
        except Exception as e:
            print(e)

    def similarity_search(self, query: str, k: int = 3) -> list:
        """Performs similarity search on the indexed documents.

        Args:
            query: The query text.
            k: The number of results to retrieve.

        Returns:
            A list of Document objects that are most similar to the query.
        """
        return self.vector_store.similarity_search(query, k=k)
