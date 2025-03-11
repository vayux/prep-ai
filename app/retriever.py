import os
import logging
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from app.embeddings import embeddings
from app.config import FAISS_INDEX_PATH, DATA_FILE

logging.basicConfig(level=logging.INFO)


class Retriever:
    def __init__(self):
        self.vectorstores = {}

    def load_documents(self, data_file):
        """Load documents from the specified data file."""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file '{data_file}' not found.")
        loader = TextLoader(data_file)
        return loader.load()

    def split_texts(self, documents):
        """Split documents into chunks for better retrieval."""
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(documents)

    def create_faiss_index(self, agent_type, data_file, index_path):
        """Create a FAISS index for a specific agent and save it."""
        logging.info(f"Creating FAISS index for {agent_type}...")

        documents = self.load_documents(data_file)
        texts = self.split_texts(documents)

        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(index_path)
        self.vectorstores[agent_type] = vectorstore

        logging.info(f"✅ {agent_type} FAISS index created successfully at {index_path}!")

    def load_faiss_index(self, agent_type, index_path):
        """Load a FAISS index for a specific agent."""
        if os.path.exists(index_path):
            self.vectorstores[agent_type] = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            return self.vectorstores[agent_type].as_retriever(search_type="similarity", search_kwargs={"k": 3})
        else:
            raise FileNotFoundError(f"FAISS index for {agent_type} not found. Please create the index first.")

    def retrieve_for_agent(self, agent_type, query):
        """Retrieve relevant information for a specific agent."""
        if agent_type not in self.vectorstores:
            raise ValueError(f"Retriever for {agent_type} is not initialized.")

        retriever = self.vectorstores[agent_type].as_retriever(search_type="similarity", search_kwargs={"k": 3})
        results = retriever.get_relevant_documents(query)
        
        return [doc.page_content for doc in results] if results else ["No relevant data found."]


# 🔥 Initialize retriever and create separate indexes for each AI agent
if __name__ == "__main__":
    retriever = Retriever()

    agent_data_files = {
        "DSA": "data/dsa_data.md",
        "LLD": "data/lld_data.md",
        "HLD": "data/hld_data.md",
        "Behavioral": "data/behavioral_data.md",
    }

    agent_indexes = {
        "DSA": "faiss_index/dsa",
        "LLD": "faiss_index/lld",
        "HLD": "faiss_index/hld",
        "Behavioral": "faiss_index/behavioral",
    }

    for agent, data_file in agent_data_files.items():
        retriever.create_faiss_index(agent, data_file, agent_indexes[agent])
