# Backend

This folder contains all backend code and assets for the Drug–Drug Interaction prototype.

## Setup

1. Python
   - Use the virtual environment in `backend/.venv` or create your own.
   - Dependencies are listed in `backend/requirements.txt`.

2. Environment variables
   - Put secrets in `backend/.env` (auto-loaded). For example:
     ```
     OPENAI_API_KEY=your_key_here
     ```

3. Data
   - Local corpus lives at `backend/data/processed/fda_documents.jsonl`.
   - To build the vector store:
     ```bash
     python -m backend.index.create_vectorstore
     ```

## Run API

- FastAPI app:
  ```bash
  uvicorn backend.main:app --reload
  ```

## Notes
- Vector store persists in `drug_vector_db/` relative to repo root unless configured otherwise.
- The pipeline entry is `backend/pipeline_full.py`.
