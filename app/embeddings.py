"""
This module initializes the SentenceTransformerEmbeddings used for embedding text data.

The embeddings are based on the 'all-MiniLM-L6-v2' model, which is suitable for various NLP tasks.
"""

from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
