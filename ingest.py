import fitz
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle

# Load PDF
doc = fitz.open("data/Hartshorne.pdf")
text = ""
for page in doc:
    text += page.get_text()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_text(text)

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "vector_store/Hartshorne.index")

# Save original chunks
with open("vector_store/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Book successfully indexed.")
