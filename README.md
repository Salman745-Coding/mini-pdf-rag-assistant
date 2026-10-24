Mini PDF Knowledge Assistant — RAG Project

A beginner-friendly Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its contents using Python, OpenAI APIs, NumPy, and Streamlit.

Project Overview

This project demonstrates how a RAG pipeline retrieves relevant information from a document and uses an AI language model to generate answers grounded in the retrieved passages.

Key Features

Upload text-based PDF documents.

Extract document text and retain PDF page numbers.

Split text into overlapping chunks.

Generate vector embeddings using OpenAI's text-embedding-3-small model.

Retrieve relevant passages using cosine similarity and NumPy.

Generate context-grounded answers using the OpenAI Responses API.

Display retrieved passages and source page numbers.

Provide an interactive chatbot interface using Streamlit.

Load API credentials from environment variables.

Technology Stack

Language: Python

AI API: OpenAI API

Embedding Model: text-embedding-3-small

Generation Model: Configurable through OPENAI_MODEL

PDF Processing: PyMuPDF

Vector Similarity: NumPy

User Interface: Streamlit

Configuration: python-dotenv

Version Control: Git and GitHub

Architecture

PDF Upload: The user uploads a PDF through the Streamlit interface.

Text Extraction: PyMuPDF extracts text and page numbers.

Chunking: The extracted text is split into chunks with overlapping content.

Embedding Generation: OpenAI converts the chunks into numerical vectors.

Retrieval: The user's question is embedded and compared with document vectors using cosine similarity.

Context Construction: The most relevant passages are selected.

Answer Generation: The OpenAI model answers using the retrieved document context.

Source Display: The application displays the answer and retrieved passages with page references.

Project Structure

mini-RAG-Project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env                 # Local only; do not commit
└── .venv/               # Local Python environment; do not commit

Getting Started

1. Clone the repository

git clone https://github.com/Salman745-Coding/mini-pdf-rag-assistant.git
cd mini-pdf-rag-assistant

2. Create and activate a virtual environment

On Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install -r requirements.txt

4. Configure your API key

Create a local file named .env in the project root:

OPENAI_API_KEY=your_actual_api_key_here
OPENAI_MODEL=gpt-4.1-mini

Replace the placeholder with your own API key. Use a generation model available to your OpenAI account.

Security: Never upload .env or expose your API key in GitHub commits.

5. Run the application

python -m streamlit run app.py

Open the local Streamlit address displayed in the terminal, usually http://localhost:8501.

Limitations

Designed for PDFs containing selectable text; scanned PDFs may require OCR.

Retrieval uses in-memory NumPy similarity search rather than a persistent vector database.

Embedding and answer generation require API access and may incur charges.

The quality of answers depends on PDF text quality, chunking, retrieval relevance, and the language model.

Future Improvements

Add OCR support for scanned documents.

Persist document embeddings using a vector database.

Add automated tests and structured error handling.

Improve retrieval using metadata filtering and reranking.

Deploy the application to a cloud hosting platform.

Author

Salman Riaz
Cybersecurity Engineer | AI and RAG Project Development

GitHub: https://github.com/Salman745-Coding
