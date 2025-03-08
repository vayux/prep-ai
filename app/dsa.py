import os
import logging
from concurrent.futures import ThreadPoolExecutor

from langchain.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS index with GPU support
try:
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        index = vectorstore.index  # Extract FAISS index

        # Check for GPU availability and move FAISS index to GPU if available
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()  # GPU resources
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU
            vectorstore.index = gpu_index
            logging.info("FAISS is running on GPU.")
        else:
            logging.warning("No GPU available, running FAISS on CPU.")

        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
    else:
        raise FileNotFoundError("FAISS index not found. Please create the index first.")
except FileNotFoundError as e:
    logging.error(e)
    raise
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    raise

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3:latest"), chain_type="stuff", retriever=retriever
)

# Query Functions
# Define functions to query the chatbot.


def query_rag(question):
    """
    Query the Retrieval-Augmented Generation (RAG) model.

    Parameters:
    question (str): The question to query.

    Returns:
    str: The response from the RAG model.
    """
    response = qa_chain.run(question)
    return response
