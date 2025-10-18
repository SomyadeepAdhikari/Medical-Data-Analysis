# Medical Data Analysis RAG – Detailed Overview

## What this project is
A local-friendly Retrieval-Augmented Generation (RAG) system over a medical-style dataset. It ingests a parquet file, builds a vector store using open-source embeddings, retrieves relevant chunks for a user question, and then generates grounded answers with a HuggingFace LLM. It also provides an optional web UI via Streamlit.

## RAG Type Used
- Type: Classical retrieve-then-read RAG
- Details: Single-stage vector similarity retrieval (Chroma) → prompt a causal LLM with the question + retrieved context.
- Retrieval strategy: Dense vector similarity (k=3 by default) using Sentence-Transformers or FastEmbed embeddings.
- No re-ranking or multi-hop chain is applied by default (can be added later as an enhancement).

## Key Components
- Dataset: `documents/train.parquet` (primary). `documents/test.parquet` exists for experiments; `gemini_dataset.txt` is legacy and not used by ingestion.
- Vector store: Chroma, persisted at `db/chroma_db` (contains `chroma.sqlite3` and binary segment files).
- Embeddings: HuggingFace Sentence-Transformers (`sentence-transformers/all-MiniLM-L6-v2`) or `FastEmbedEmbeddings` if available (torch-free).
- LLM: HuggingFace `transformers` text-generation pipeline. Defaults to a small, CPU-friendly model and can be swapped via env var.
- Optional translation: Sarvam AI for translating non-English questions to English and back.
- Interfaces: CLI (`Rag_final.py`, `one_to_one_question_rag.py`), ingestion (`Rag_1.py`), and Streamlit UI (`streamlit_app.py`).

## Data Flow
1. Ingestion (Rag_1.py)
   - Reads `documents/train.parquet` with pandas.
   - Builds a textual representation per row using object/string columns (fallback: all columns).
   - Splits documents with `CharacterTextSplitter(chunk_size=750, chunk_overlap=100)`.
   - Creates embeddings:
     - Prefers `FastEmbedEmbeddings` if installed (avoids torch), else `HuggingFaceEmbeddings` with `HUGGINGFACE_EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`).
   - Persists chunks + vectors into Chroma at `db/chroma_db`.

2. Retrieval + Generation (Rag_final.py)
   - Loads the persisted Chroma DB with the same embedding function.
   - Optional language detection + translation:
     - If query is not English, translates to English via Sarvam (if `SARVAM_API_KEY` is set).
   - Retrieves top-k similar chunks (k=3).
   - Builds a grounded prompt that includes the question and the retrieved context, with an instruction to answer strictly from the documents or say "I'm not sure."
   - Lazy-loads a HuggingFace LLM pipeline for generation (GPU if `HUGGINGFACE_DEVICE=cuda`, else CPU). Has fallbacks to a tiny model if loading fails or is too large for CPU.
   - If original query language was not English, translates the final answer back to the original language (if Sarvam is configured).

## Scripts and Their Roles
- `Rag_1.py` – Build/refresh the vector store from the parquet dataset. Honors `REBUILD_DB` and `INGEST_MAX_ROWS`.
- `Rag_final.py` – The main RAG engine:
  - Interactive mode (default if no `--query`): a chat-like loop in the terminal.
  - Single-shot mode via `--query`.
  - Handles language detection/translation, retrieval, generation, and fallback behavior.
- `one_to_one_question_rag.py` – Simplified single-question runner (legacy; `Rag_final.py --query` supersedes it).
- `streamlit_app.py` – Streamlit UI with dark theme and a sidebar to load the model, clear chat, and adjust retrieval `k` (display-only today).

## Configuration (Environment Variables)
- Embeddings
  - `HUGGINGFACE_EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- LLM / Generation
  - `HUGGINGFACE_LLM_MODEL` (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
  - `HUGGINGFACE_DEVICE` = `cpu` or `cuda`
  - `LLM_TEMPERATURE` (default: `0.2`)
  - `ALLOW_LARGE_CPU_MODEL` = `true|false` (default false; prevents loading large models on CPU)
  - `SKIP_LLM` = `true|false` (retrieval-only mode)
- Translation
  - `SARVAM_API_KEY`
  - `SARVAM_BASE_URL` (default: `https://api.sarvam.ai`)
- Ingestion
  - `REBUILD_DB` = `true|false` (force rebuild)
  - `INGEST_MAX_ROWS` = `0` for no limit or an integer cap
- Optional
  - `SAMPLE_QUERY` for `one_to_one_question_rag.py`
  - `HUGGINGFACEHUB_API_TOKEN` if a model requires authentication

## Dependencies (from `requirements.txt`)
- Core RAG: `langchain`, `langchain-community`, `langchain-chroma`, `chromadb`, `python-dotenv`
- HuggingFace & NLP: `transformers`, `sentence-transformers`, `huggingface-hub`, `torch`, `accelerate`, `safetensors`
- Embedding fallback: `fastembed`
- Data: `pandas`, `numpy`, `pyarrow`
- UI: `streamlit`
- Utilities: `requests`
- Optional: `langdetect`

## How to Run (Windows PowerShell)
1) Create venv and install dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

2) Build the vector store
```powershell
python .\Rag_1.py
# or via VS Code task: Tasks → Run Task → "Build vector store"
```

3) Run the CLI (interactive)
```powershell
python .\Rag_final.py
```

Single question
```powershell
python .\Rag_final.py --query "What are the main findings about Coronary Artery Disease in the dataset?"
```

4) Run the Streamlit UI
```powershell
streamlit run .\streamlit_app.py
```

## Operational Tips
- Small/fast model for smoke tests: `sshleifer/tiny-gpt2` (set `HUGGINGFACE_LLM_MODEL` accordingly).
- CPU-only: keep `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or similar small instruct models to avoid OOM.
- If you see retrieval-only behavior, check `SKIP_LLM`, model load logs, or pick a smaller model.
- If embeddings fail due to torch issues, installing `fastembed` helps avoid torch for embeddings.

## Streamlit UX Notes
- Dark theme configured in `.streamlit/config.toml`.
- Sidebar: load model, clear chat, display current model, and a `k` slider (retriever currently defaults to 3 in the backend).
- Context Tab: inspect the raw retrieved text used for the last answer.

## Future Enhancements
- Pass sidebar `k` directly to retrieval calls.
- Add a re-ranking step or multi-hop retrieval for more complex questions.
- Improve language detection (e.g., fastText) and robust translation error handling.
- Provide a lightweight FastAPI endpoint for programmatic access.
- Add relevance/evaluation metrics and unit tests for ingestion and RAG steps.

---
This document complements `Readme.md` with an implementation-focused view and explicitly states the RAG type: classical retrieve-then-read with dense vector similarity search using Chroma.
