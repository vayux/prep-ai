"""
This module handles the retrieval of documents and their processing for use in AI models.

It includes functions for loading documents, splitting text into chunks, and managing FAISS vector stores.
"""

import os
import logging

from langchain.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


# Configure logging
logging.basicConfig(level=logging.INFO)

# Ensure the data file exists
try:
    DATA_FILE = "data/DSA.md"
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Data file '{DATA_FILE}' not found. Please add your dataset."
        )
except FileNotFoundError as e:
    logging.error(e)
    raise

# Load documents


def load_documents(file_path):
    """
    Load documents from the specified file path.

    Parameters:
    file_path (str): The path to the data file.

    Returns:
    list: A list of loaded documents.
    """
    loader = TextLoader(file_path)
    return loader.load()


# Split text into chunks


def split_texts(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into chunks of specified size and overlap.

    Parameters:
    documents (list): The list of documents to split.
    chunk_size (int): The size of each chunk.
    chunk_overlap (int): The overlap between chunks.

    Returns:
    list: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


# Initialize embeddings


def initialize_embeddings(model_name="all-MiniLM-L6-v2"):
    """
    Initialize sentence transformer embeddings.

    Parameters:
    model_name (str): The name of the model to use for embeddings.

    Returns:
    SentenceTransformerEmbeddings: The initialized embeddings.
    """
    return SentenceTransformerEmbeddings(model_name=model_name)


# Query LLM
def query_llm(question):
    """Unified function to query LLM"""
    llm = Ollama(model="llama3:latest")
    try:
        response = llm.invoke(question)
        return response if isinstance(response, str) else str(response)
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        return "⚠️ Error processing LLM request."


# Create and save FAISS index
def create_faiss_index(texts, embeddings, index_path="faiss_index"):
    """
    Create a FAISS index from the given texts and embeddings, and save it locally.

    Parameters:
    texts (list): The list of text chunks.
    embeddings (SentenceTransformerEmbeddings): The embeddings to use.
    index_path (str): The path to save the FAISS index.
    """
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(index_path)
    logging.info("✅ FAISS index created successfully!")


# Main execution
if __name__ == "__main__":
    try:
        documents = load_documents(DATA_FILE)
        texts = split_texts(documents)
        embeddings = initialize_embeddings()
        create_faiss_index(texts, embeddings)
    except Exception as e:
        logging.error(f"Error in creating FAISS index: {e}")
        raise
