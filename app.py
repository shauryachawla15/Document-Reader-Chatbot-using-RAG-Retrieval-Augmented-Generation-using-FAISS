from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pypdf import PdfReader
import faiss
import numpy as np
import os

# ------------------------
# Setup
# ------------------------
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("API KEY FOUND:", bool(os.getenv("OPENAI_API_KEY")))

# Local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# Load PDF
# ------------------------
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

text = load_pdf_text("data/s3.pdf")

# ------------------------
# Chunking
# ------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

chunks = chunk_text(text)

# ------------------------
# Embeddings + FAISS Index
# ------------------------
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

print(f"FAISS index built with {index.ntotal} vectors")

# ------------------------
# Routes
# ------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please ask a valid question."})

    # Embed question
    question_embedding = embedder.encode(
        [question], convert_to_numpy=True
    )

    # Search FAISS (top 3 chunks)
    k = 3
    distances, indices = index.search(question_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the document, say "Not found in the document."

Context:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    answer = response.output_text.strip()
    return jsonify({"answer": answer})

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=False)
