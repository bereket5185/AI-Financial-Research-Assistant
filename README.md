# AI Financial Research Assistant

A Streamlit application for researching financial documents with semantic search and AI-assisted analysis. The app processes annual reports, finds relevant passages with vector embeddings, and presents research results through an authenticated web interface.

## Features

- Upload and process financial documents such as annual reports
- Semantic search over processed document chunks
- AI-assisted research using a local Ollama model
- User authentication with user and administrator roles
- Saved research history, sources, and extracted metrics
- SQLite database for users, documents, and research records
- Streamlit interface for interactive exploration

## Project Structure

- `app/app.py` - Main Streamlit application
- `database/db.py` - SQLite database setup and data access functions
- `data/documents/` - Source financial documents
- `data/processed/` - Processed text chunks, page data, and embeddings
- `notebooks/` - Exploratory document-processing and search notebooks
- `preprocess_multi_documents.py` - Document preprocessing script

## Setup

1. Create and activate a Python virtual environment.
2. Install the required Python packages used by the application, including Streamlit, pandas, NumPy, scikit-learn, sentence-transformers, matplotlib, and Ollama's Python client.
3. Install and start [Ollama](https://ollama.com/), then download the model configured by the application.
4. Make sure the required processed documents and embedding files are available under `data/processed/`.

## Run the Application

From the project root:

```bash
streamlit run app/app.py
```

The local SQLite database is created automatically at `data/financial_ai.db` when the application starts. This generated database is intentionally excluded from version control.

## Notes

The application expects a locally available Ollama service for AI-assisted responses. Model names and document-processing settings can be adjusted in `app/app.py` and the preprocessing scripts.
