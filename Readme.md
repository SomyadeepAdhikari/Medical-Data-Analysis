# Medical Data Analysis RAG – Project Summary

## 1. Overview
This project implements a Retrieval-Augmented Generation (RAG) workflow over a medical-style text dataset using a fully free/open-source stack (HuggingFace + Chroma) with optional multilingual translation via Sarvam AI.

Originally it used Google Generative AI (Gemini) for embeddings + chat. All paid / proprietary Google APIs have been removed and replaced with:
- Embeddings: `HuggingFaceEmbeddings` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- Generation: HuggingFace causal LLM pipeline (defaults evolved from a large model to a lighter one for local usage)
- Translation (optional): Sarvam API scaffold

## 2. Key Components
| Layer | Technology | Notes |
|-------|------------|-------|
| Vector Store | Chroma | Persisted under `db/chroma_db` |
| Embeddings | sentence-transformers | Configurable via env var |
| LLM | HuggingFace causal LM | Lazy-loaded with fallback logic |
| Translation (optional) | Sarvam AI | Bidirectional for multilingual queries |

## 3. Files Modified / Added
| File | Purpose | Key Changes |
|------|---------|-------------|
| `Rag_1.py` | Ingestion + embedding + persistence | Switched to HF embeddings; env override for model name |
| `Rag_final.py` | Interactive retrieval + generation loop | Removed Gemini, added HF pipeline, lazy loading, fallback, translation scaffold, diagnostics |
| `one_to_one_question_rag.py` | Single prompt RAG answer | Migrated to HF embeddings + generation; translation support |
| `requirements.txt` | Dependencies | Added transformers stack; removed Google AI SDKs; added torch/accelerate/safetensors |
| `.env.example` | Config template | New variables for HF + Sarvam + optional query override |
| `.gitignore` | Hygiene | Ignore venv, db, .env, caches, parquet |
| `PROJECT_SUMMARY.md` | Documentation | (This document) |

## 4. Environment Variables
```
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
HUGGINGFACE_DEVICE=cpu         # or cuda
LLM_TEMPERATURE=0.2
SKIP_LLM=false                 # set true for retrieval-only mode
SARVAM_API_KEY=your_sarvam_key_here
SARVAM_BASE_URL=https://api.sarvam.ai
# SAMPLE_QUERY=What are the main findings about Coronary Artery Disease in the dataset?
```
Other optional:
```
HUGGINGFACEHUB_API_TOKEN=your_hf_token_if_needed
```

## 5. Installation & Setup
Create / activate venv (already done, recap):
```
python -m venv venv
./venv/Scripts/activate    # PowerShell
pip install -r requirements.txt
```
(If torch fails, install a platform-specific wheel from https://pytorch.org/.)

## 6. Building / Updating the Vector Store
Run when the dataset changes or first time:
```
python Rag_1.py
```
This loads `Documents/train.parquet`, splits it, embeds chunks, and persists to `db/chroma_db`.

## 7. Running RAG
Interactive loop:
```
python Rag_final.py
```
Exit with `exit`.

Single query:
```
python one_to_one_question_rag.py
```
Override query:
```
setx SAMPLE_QUERY "List patients with fatigue"
```
(or add to `.env`).

## 8. Retrieval-Only Debug Mode
To inspect what the retriever pulls without generating:
```
SKIP_LLM=true python Rag_final.py
```
(or set `SKIP_LLM=true` in `.env`).

## 9. Multilingual Flow (Optional Sarvam)
1. Detect language (naive ASCII heuristic)
2. If not English → translate to English via Sarvam
3. Perform retrieval + generation
4. Translate answer back to original language

`SARVAM_API_KEY` must be present; otherwise translation functions no-op.

## 10. Model Selection & Performance Tips
| Scenario | Recommended Model |
|----------|-------------------|
| Low RAM / CPU-only | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Better quality (still moderate) | `microsoft/Phi-3.5-mini-instruct` |
| Larger instruct (needs GPU) | `mistralai/Mistral-7B-Instruct-v0.3` |
| Smoke test of pipeline | `sshleifer/tiny-gpt2` |

Switch by editing `.env` or exporting `HUGGINGFACE_LLM_MODEL`.

## 11. Fallback & Resilience Logic (Interactive Script)
- Lazy load only on first generation request
- If load fails → fallback to `distilgpt2`
- If fallback fails → retrieval-only mode with warning
- `SKIP_LLM=true` bypasses model entirely
- Response trimmed to 4000 chars to avoid runaway output

## 12. Diagnostics Added
- Prints torch version & CUDA availability
- Advises on choosing smaller models if OOM occurs
- Encourages using `cpu` or lighter architectures

## 13. Security / Hygiene
- Removed commented API key
- `.env` excluded from version control
- Vector store and parquet files ignored

## 14. Testing Performed
- Imports validated (`transformers`, `sentence_transformers`, `huggingface_hub`, `requests`, `langchain.schema`)
- Tiny generation test with `sshleifer/tiny-gpt2`
- Verified ingestion path still constructs & persists Chroma DB

## 15. Known Limitations / Next Opportunities
| Area | Opportunity |
|------|-------------|
| Language detection | Replace ASCII heuristic with `langdetect` or fastText |
| Translation API | Adjust payload to real Sarvam spec + error classification |
| Prompt handling | Maintain structured history or token window trimming |
| Evaluation | Add retrieval quality metrics or relevance scoring report |
| Serving | Wrap in FastAPI or Gradio for UI/REST access |
| Caching | Add translation + generation caching (e.g., simple SQLite) |
| Streaming | Implement token streaming for larger models |

## 16. Quick Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| Slow startup | Large model download/load | Switch to TinyLlama / tiny-gpt2 |
| OOM / crash | Model too big for RAM/GPU | Set `HUGGINGFACE_LLM_MODEL=TinyLlama/...` and `HUGGINGFACE_DEVICE=cpu` |
| Blank / odd answers | Model not instruction-tuned | Use instruct/chat model (Phi, Mistral, Zephyr) |
| No translation | Missing Sarvam key | Set `SARVAM_API_KEY` |
| Only retrieval prints | LLM skipped or load failed | Check logs, disable `SKIP_LLM`, verify model name |

## 17. Minimal End-to-End Test Script (Ad Hoc)
```
# 1. Build vector store
python Rag_1.py
# 2. Run interactive with tiny model
setx HUGGINGFACE_LLM_MODEL sshleifer/tiny-gpt2
python Rag_final.py
# 3. Ask: "What are the main findings about Coronary Artery Disease in the dataset?"
```

## 18. Summary
You now have a fully local-friendly, configurable RAG pipeline with:
- Pluggable embeddings & models
- Multilingual scaffolding
- Retrieval-only diagnostic mode
- Resilient loading + fallbacks

> Next recommended enhancement: add a FastAPI endpoint (`/ask`) plus a `/health` route to facilitate integration.

## 19. Streamlit UI (New)
An interactive dark-themed web interface is available via `streamlit_app.py`.

### Launch
```
streamlit run streamlit_app.py
```

Ensure you have first built the vector store (run `python Rag_1.py`).

### Features
- Chat experience with fake streaming effect
- Sidebar settings: retriever k, model display, load model button, clear chat
- Retrieval-only indicator if `SKIP_LLM=true`
- Context inspection tab to view raw retrieved chunks
- Dark theme configured in `.streamlit/config.toml`
- Lazy model load only when requested (saves memory)

### Environment Influence
- `HUGGINGFACE_LLM_MODEL` shows which model is targeted
- `SKIP_LLM=true` disables generation and shows only context

### Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Empty retriever | Vector DB not built | Run `python Rag_1.py` |
| Model never loads | Large model OOM | Choose smaller model in `.env` |
| Only context, no answer | Retrieval-only mode or load failure | Check logs / disable `SKIP_LLM` |

---
*UI module is optional; core CLI scripts still function independently.*

---
*Generated documentation. Keep this file updated as you iterate.*
