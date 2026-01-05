# Document-Reader-Chatbot-using-RAG-Retrieval-Augmented-Generation-using-FAISS
A **Retrieval-Augmented Generation (RAG)** based chatbot that allows users to ask natural language questions from PDF documents.  
The system retrieves relevant document context using vector similarity search and generates accurate, grounded answers using an LLM.

---

## 🧠 Project Overview

This project implements a **RAG-based document question answering system** using:
- Local embedding models (Sentence Transformers)
- Vector similarity search (FAISS)
- OpenAI language models for response generation

Instead of relying on the LLM’s internal knowledge, the chatbot retrieves the most relevant parts of the document and answers **strictly based on the document content**, minimizing hallucinations.

---

## 🚀 Features

- 📄 PDF text extraction
- ✂️ Document chunking
- 🧠 Semantic embeddings (local & free)
- 🔍 FAISS vector search
- 🤖 LLM-based answer generation
- 🛡️ Context-restricted answers (RAG)
- 🌐 Flask web interface
- ⚡ Fast and lightweight

---

## 🏗️ Architecture (How It Works)

1. Load and extract text from a PDF
2. Split text into fixed-size chunks
3. Convert chunks into embeddings
4. Store embeddings in a FAISS index
5. Embed user question
6. Retrieve the most relevant chunk
7. Generate answer using the retrieved context

---

## 🛠️ Tech Stack

| Component | Technology |
|--------|-----------|
| Backend | Flask |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS |
| LLM | OpenAI (`gpt-4o-mini`) |
| PDF Parsing | PyPDF |
| Frontend | HTML, CSS |
| Language | Python |

---

## 📂 Project Structure

```text
doc-reader-chatbot/
│
├── app.py
├── requirements.txt
├── data/
│   └── sample.pdf
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md



