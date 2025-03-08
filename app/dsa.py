from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import os
import logging
import faiss
from langchain.chains import RetrievalQA
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS index with GPU support
try:
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        index = vectorstore.index  # Extract FAISS index

        # Check for GPU availability and move FAISS index to GPU if available
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()  # GPU resources
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU
            vectorstore.index = gpu_index
            logging.info("FAISS is running on GPU.")
        else:
            logging.warning("No GPU available, running FAISS on CPU.")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    else:
        raise FileNotFoundError("FAISS index not found. Please create the index first.")
except FileNotFoundError as e:
    logging.error(e)
    raise
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    raise

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3:latest"),
    chain_type="stuff",
    retriever=retriever
)

# Query Functions
# Define functions to query the chatbot.
def query_rag(question):
    response = qa_chain.run(question)
    return response

def query_llm_sync(question):
    llm = Ollama(model="llama3:latest")
    try:
        response = llm.invoke(question)
        return response if isinstance(response, str) else str(response)
    except Exception as e:
        return f"Error: {e}"

# Function to directly query LLM (without retrieval)
def query_llm_directly(user_query):
    """
    Directly query the LLM without retrieval.

    Parameters:
    user_query (str): The user's query.

    Returns:
    str: The LLM's response.
    """
    try:
        llm = Ollama(model="mistral")
        return llm(user_query).strip()
    except Exception as e:
        logging.error(f"Error in query_llm_directly: {e}")
        return "An error occurred while querying the LLM."

def query_rag_and_llm_parallel(question):
    with ThreadPoolExecutor() as executor:
        rag_future = executor.submit(qa_chain.run, question)
        llm_future = executor.submit(query_llm_sync, question)
        rag_response = rag_future.result()
        llm_response = llm_future.result()
    return rag_response, llm_response