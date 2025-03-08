from langchain_community.llms import Ollama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def query_llm_sync(question):
    """
    Query the LLM synchronously with the given question.

    Parameters:
    question (str): The question to query the LLM with.

    Returns:
    str: The response from the LLM.
    """
    llm = Ollama(model="llama3:latest")
    try:
        response = llm.invoke(question)
        return response if isinstance(response, str) else str(response)
    except Exception as e:
        logging.error(f"Error in query_llm_sync: {e}")
        raise
