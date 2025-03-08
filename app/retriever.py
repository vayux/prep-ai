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
        self.vectorstore = None

    def load_documents(self):
        """Load documents from file."""
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Data file '{DATA_FILE}' not found.")
        loader = TextLoader(DATA_FILE)
        return loader.load()

    def split_texts(self, documents):
        """Split documents into chunks."""
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(documents)

    def create_faiss_index(self, texts):
        """Create FAISS index and save it."""
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        self.vectorstore = vectorstore
        logging.info("✅ FAISS index created successfully!")

    def load_faiss_index(self):
        """Load FAISS index if it exists."""
        if os.path.exists(FAISS_INDEX_PATH):
            self.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        else:
            raise FileNotFoundError("FAISS index not found. Please create the index first.")


if __name__ == "__main__":
    retriever = Retriever()
    documents = retriever.load_documents()
    texts = retriever.split_texts(documents)
    retriever.create_faiss_index(texts)
