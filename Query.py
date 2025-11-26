import faiss
import pickle
from sentence_transformers import SentenceTransformer
import subprocess

index = faiss.read_index("vector_store/Hartshorne.index")

with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')


def retrieve(query, k=5):
    q_embedding = model.encode([query])
    distances, indices = index.search(q_embedding, k)
    return [chunks[i] for i in indices[0]]


def ask(query):
    context = "\n".join(retrieve(query))

    prompt = f"""
You are an Algebraic Geometry tutor. Use ONLY the following textbook content.

Context:
{context}

Question: {query}
Answer with clear derivations.
"""

    result = subprocess.run(
        ["ollama", "run", "tinyllama"],
        input=prompt,
        text=True,
        capture_output=True
    )

    print(result.stdout)


while True:
    q = input("Ask AG: ")
    ask(q)
