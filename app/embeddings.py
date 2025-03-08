from langchain.embeddings import SentenceTransformerEmbeddings
from app.config import EMBEDDING_MODEL

embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
