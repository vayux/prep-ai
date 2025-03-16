import numpy as np
import faiss

# Check NumPy version
print(np.__version__)

# FAISS test
dim = 128  # Vector dimension
nb = 1000  # Number of vectors
np.random.seed(1234)
xb = np.random.random((nb, dim)).astype('float32')

index = faiss.IndexFlatL2(dim)  # Using a flat (brute-force) index
index.add(xb)  # Add vectors to the index
print(f"Number of vectors in the index: {index.ntotal}")
