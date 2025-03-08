from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Ensure the data file exists
try:
    data_file = "data/DSA.md"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found. Please add your dataset.")
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
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
        documents = load_documents(data_file)
        texts = split_texts(documents)
        embeddings = initialize_embeddings()
        create_faiss_index(texts, embeddings)
    except Exception as e:
        logging.error(f"Error in creating FAISS index: {e}")
        raise

