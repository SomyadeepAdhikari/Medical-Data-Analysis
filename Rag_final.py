import os
import json
import argparse
import tempfile
import requests
from dotenv import load_dotenv
import importlib.util

# --- NumPy compatibility shims for NumPy 2.x (used by some deps like chromadb) ---
try:
    import numpy as _np  # type: ignore
    # Some older libs reference np.float_ / np.uint aliases removed in NumPy 2
    if not hasattr(_np, "float_"):
        _np.float_ = _np.float64  # type: ignore[attr-defined]
    if not hasattr(_np, "uint"):
        _np.uint = _np.uint32  # type: ignore[attr-defined]
except Exception:
    pass

from langchain_chroma import Chroma
from chromadb.config import Settings  # disable telemetry
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_community.embeddings import FastEmbedEmbeddings  # type: ignore
    _FASTEMBED = True
except Exception:
    _FASTEMBED = False
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import importlib
from typing import Optional, List

# Optional fallback translator (no API key required)
try:
    from deep_translator import GoogleTranslator  # type: ignore
    _DEEPLITE_AVAILABLE = True
except Exception:
    _DEEPLITE_AVAILABLE = False

# Hybrid retrieval additions
try:
    from langchain_community.retrievers import BM25Retriever  # type: ignore
    from langchain.retrievers import EnsembleRetriever  # type: ignore
    _HYBRID_AVAILABLE = True
except Exception:
    _HYBRID_AVAILABLE = False

try:
    # Document class for rebuilding BM25 docs from Chroma
    from langchain_core.documents import Document  # type: ignore
except Exception:
    Document = None  # type: ignore

# Optional better language detection (langdetect) fallback to heuristic
try:
    from langdetect import detect  # type: ignore
    _LANGDETECT_AVAILABLE = True
except Exception:
    _LANGDETECT_AVAILABLE = False

# Diagnostics: verify torch availability early
try:
    import importlib.util as _importlib_util
    if _importlib_util.find_spec("torch") is not None:
        import torch  # type: ignore
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"[INFO] torch version: {torch.__version__}; CUDA available: {torch.cuda.is_available()}; MPS available: {mps_ok}")
    else:
        print("[WARNING] torch not installed. Install with: pip install torch --upgrade")
except Exception:
    print("[WARNING] torch detection failed.")

# Load environment variables
load_dotenv()

# ---------------- Hugging Face Inference API (Hosted) ----------------
# Some environments (notably Python 3.13 on macOS) may not have PyTorch wheels
# available, making local Transformers generation impossible. In that case we
# can fall back to Hugging Face hosted inference, with a configurable base URL.

def _truthy_env(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _hf_token() -> str:
    # Support the common HF env var names.
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_API_TOKEN")
        or ""
    )
    # Be tolerant of accidentally quoted values.
    token = token.strip().strip('"').strip("'")
    return token


def _hf_inference_base_urls() -> list[str]:
    """Return candidate base URLs to try for HF hosted inference.

    HF has migrated/introduced routing endpoints over time; trying a small
    fallback list makes the project resilient to base URL changes.
    """
    override = (
        os.getenv("HF_INFERENCE_BASE_URL")
        or os.getenv("HUGGINGFACE_INFERENCE_BASE_URL")
        or os.getenv("HUGGINGFACEHUB_INFERENCE_BASE_URL")
        or ""
    ).strip().rstrip("/")
    if override:
        # Users sometimes paste a full endpoint ending with "/models".
        # Normalize to the base so we can safely append "/models/{repo_id}".
        if override.endswith("/models"):
            override = override[: -len("/models")]
        return [override]
    return [
        "https://api-inference.huggingface.co",
        "https://router.huggingface.co/hf-inference",
    ]


def _hf_hosted_generate(prompt: str, *, model: str, max_new_tokens: int = 512) -> str:
    """Generate text using Hugging Face hosted inference.

    Requires an access token for gated models and for higher rate limits.
    """
    token = _hf_token()
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN in .env to use hosted inference."
        )

    try:
        from huggingface_hub import InferenceClient  # type: ignore
    except Exception as e:
        raise RuntimeError(f"huggingface_hub is required for hosted inference but could not be imported: {e}")

    last_err: Exception | None = None
    for base_url in _hf_inference_base_urls():
        try:
            # In huggingface_hub<=0.24.x, the InferenceClient does not expose a
            # separate "inference endpoint base" argument. The supported way to
            # target a different base URL is to pass a full URL as the model.
            model_url = f"{base_url.rstrip('/')}/models/{model}"
            client = InferenceClient(model=model_url, token=token, timeout=90)
            out = client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
            if isinstance(out, str):
                return out.strip()
            # Fallback: if a typed object is returned for some reason.
            generated = getattr(out, "generated_text", None) or getattr(out, "text", None)
            if isinstance(generated, str) and generated.strip():
                return generated.strip()
            # If we can't interpret the return value, treat as an error.
            raise RuntimeError(f"Unexpected hosted inference response type: {type(out)}")
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Hosted inference failed for all base URLs. Last error: {last_err}")

# Define the path to the Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

"""
RAG Interactive QA Script (Free Stack)

Changes:
- Removed Google Generative AI usage.
- Uses HuggingFace embeddings (default: sentence-transformers/all-MiniLM-L6-v2).
- Uses a HuggingFace causal LM for generation (default: mistralai/Mistral-7B-Instruct-v0.3; change via env HUGGINGFACE_LLM_MODEL).
- Adds optional Sarvam AI translation for multilingual Q&A if SARVAM_API_KEY is present.

Environment Variables (update your .env):
    HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    HUGGINGFACE_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
    HUGGINGFACE_DEVICE=cpu  # or cuda
    LLM_TEMPERATURE=0.2
    SARVAM_API_KEY=your_key_here
    SARVAM_BASE_URL=https://api.sarvam.ai
"""

def _build_dense_retriever(db: Chroma, k: int):
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k})


def _build_hybrid_retriever(db: Chroma, k: int, dense_weight: float = 0.6) -> object:
    """Create a hybrid retriever (dense + BM25) if available, else return dense-only retriever.

    dense_weight controls the blend; lexical weight = (1 - dense_weight).
    """
    dense = _build_dense_retriever(db, k)
    if not _HYBRID_AVAILABLE or Document is None:
        print("[INFO] Hybrid retrieval not available. Falling back to dense-only.")
        return dense
    try:
        # Fetch all documents from Chroma to build a BM25 index in-memory
        raw = db.get(include=["documents", "metadatas"])  # type: ignore[attr-defined]
        docs_texts: List[str] = raw.get("documents", []) or []
        metas: List[dict] = raw.get("metadatas", []) or []
        all_docs: List[Document] = []  # type: ignore[assignment]
        for i, t in enumerate(docs_texts):
            if not t:
                continue
            md = metas[i] if i < len(metas) else {}
            all_docs.append(Document(page_content=t, metadata=md))  # type: ignore[arg-type]
        if not all_docs:
            print("[WARN] No documents available for BM25. Using dense-only retriever.")
            return dense
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = k
        hybrid = EnsembleRetriever(retrievers=[dense, bm25], weights=[dense_weight, 1 - dense_weight])
        print(f"[INFO] Hybrid retriever ready (k={k}, weights: dense={dense_weight}, bm25={1-dense_weight}).")
        return hybrid
    except Exception as e:
        print(f"[WARN] Failed to initialize hybrid retriever: {e}. Using dense-only retriever.")
        return dense


def init_vector_store(mode: str = "dense", k: int = 6):
    embedding_model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = None
    if _FASTEMBED:
        try:
            print("[INFO] Using FastEmbedEmbeddings for retrieval (torch-free)")
            embeddings = FastEmbedEmbeddings()
        except Exception as fe:
            print(f"[WARN] FastEmbedEmbeddings failed: {fe}. Falling back to HuggingFaceEmbeddings: {embedding_model_name}")
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    # Initialize Chroma with graceful recovery if the on-disk DB schema is incompatible
    try:
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
            collection_name=os.getenv("CHROMA_COLLECTION", "medical_docs"),
        )
    except Exception as e:
        msg = str(e)
        # Common case: sqlite schema mismatch after library upgrades
        if ("no such column" in msg) or ("OperationalError" in msg) or ("schema" in msg and "mismatch" in msg):
            import shutil, time
            backup_dir = f"{persistent_directory}.bak-{int(time.time())}"
            try:
                print(f"[WARN] Chroma DB schema appears incompatible. Backing up and recreating: {backup_dir}")
                shutil.move(persistent_directory, backup_dir)
            except Exception as be:
                print(f"[WARN] Could not back up existing DB: {be}. Attempting fresh directory.")
                try:
                    shutil.rmtree(persistent_directory, ignore_errors=True)
                except Exception:
                    pass
            os.makedirs(persistent_directory, exist_ok=True)
            try:
                db = Chroma(
                    persist_directory=persistent_directory,
                    embedding_function=embeddings,
                    client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
                    collection_name=os.getenv("CHROMA_COLLECTION", "medical_docs"),
                )
            except Exception as e2:
                print(f"[WARN] Fresh persistent DB still failing ({e2}). Falling back to in-memory store.")
                db = Chroma(
                    embedding_function=embeddings,
                    client_settings=Settings(anonymized_telemetry=False),
                    collection_name=os.getenv("CHROMA_COLLECTION", "medical_docs"),
                )
        else:
            raise
    mode = (mode or "dense").lower()
    if mode == "hybrid":
        return _build_hybrid_retriever(db, k=k)
    # default: dense
    return _build_dense_retriever(db, k=k)

hf_model_name = os.getenv("HUGGINGFACE_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
device = os.getenv("HUGGINGFACE_DEVICE", "cpu")
temperature = float(os.getenv("LLM_TEMPERATURE", 0.2))
skip_model = os.getenv("SKIP_LLM", "false").lower() in {"1", "true", "yes"}
do_sample_default = os.getenv("LLM_DO_SAMPLE", "false").lower() in {"1", "true", "yes"}
use_hosted_inference = _truthy_env("USE_HF_INFERENCE_API", "false")
force_local_llm = _truthy_env("FORCE_LOCAL_LLM", "false")

generation_pipeline = None
parser = StrOutputParser()

def load_generation_pipeline():
    global generation_pipeline
    if generation_pipeline is not None:
        return generation_pipeline
    if skip_model:
        print("[INFO] SKIP_LLM=true -> Skipping model load. Retrieval only mode.")
        return None

    # If torch isn't installed, local Transformers generation won't work.
    # Prefer hosted inference in that case (or retrieval-only), unless forced.
    if not force_local_llm and not _torch_available():
        print("[INFO] torch not installed -> skipping local LLM load. Set USE_HF_INFERENCE_API=true for hosted inference, or FORCE_LOCAL_LLM=true to attempt local anyway.")
        return None
    # Auto-downsize safeguard for CPU when a large model is requested
    allow_large_cpu = os.getenv("ALLOW_LARGE_CPU_MODEL", "false").lower() in {"1", "true", "yes"}
    large_markers = ["mistral", "mistralai/", "mixtral", "qwen", "phi-3", "zephyr-7b", "7b", "8b", "13b", "70b"]
    if device == "cpu" and not allow_large_cpu:
        lower_name = hf_model_name.lower()
        is_tiny = ("tinyllama" in lower_name) or ("tiny" in lower_name and "gpt2" in lower_name)
        if not is_tiny and any(m in lower_name for m in large_markers):
            print(f"[WARN] {hf_model_name} appears large for CPU. Switching to TinyLlama. Set ALLOW_LARGE_CPU_MODEL=true to override.")
            # Switch to a lighter default
            os.environ["HUGGINGFACE_LLM_MODEL"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            globals()["hf_model_name"] = os.getenv("HUGGINGFACE_LLM_MODEL")
    requested_device = (device or "cpu").lower()
    resolved_device = requested_device
    torch_dtype = None
    pipeline_device = -1

    # Resolve device and check availability
    if requested_device in {"cuda", "gpu"}:
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                print("[WARN] HUGGINGFACE_DEVICE=cuda requested but CUDA not available. Falling back to CPU.")
                resolved_device = "cpu"
            else:
                resolved_device = "cuda"
        except Exception:
            print("[WARN] Torch not available for CUDA. Falling back to CPU.")
            resolved_device = "cpu"
    elif requested_device in {"mps", "apple"}:
        try:
            import torch  # type: ignore
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not mps_ok:
                print("[WARN] HUGGINGFACE_DEVICE=mps requested but MPS not available. Falling back to CPU.")
                resolved_device = "cpu"
            else:
                resolved_device = "mps"
        except Exception:
            print("[WARN] Torch not available for MPS. Falling back to CPU.")
            resolved_device = "cpu"
    else:
        resolved_device = "cpu"

    if resolved_device == "cpu":
        pipeline_device = -1
    elif resolved_device == "cuda":
        pipeline_device = 0
        torch_dtype = "auto"
    elif resolved_device == "mps":
        # Newer Transformers accepts device="mps" for pipeline.
        pipeline_device = "mps"
        torch_dtype = "auto"

    print(f"Loading HuggingFace model: {hf_model_name} on {resolved_device}")
    if resolved_device == "cuda":
        print("[INFO] If you encounter CUDA OOM, choose a smaller model or reduce max_new_tokens.")
    if resolved_device == "mps":
        print("[INFO] Using Apple Silicon MPS acceleration.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        model_kwargs = {"trust_remote_code": True}
        # device_map="auto" works well for CUDA; for MPS we move the model explicitly.
        if resolved_device == "cuda":
            model_kwargs["device_map"] = "auto"
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        model_ref = AutoModelForCausalLM.from_pretrained(hf_model_name, **model_kwargs)
        if resolved_device == "mps":
            # Ensure weights are on MPS; device_map isn't used for MPS.
            model_ref = model_ref.to("mps")
        # If we're using greedy decoding (do_sample=False), unset sampling-only knobs
        # to avoid noisy warnings from Transformers.
        if not do_sample_default:
            try:
                if hasattr(model_ref, "generation_config") and model_ref.generation_config is not None:
                    model_ref.generation_config.top_k = None
                    model_ref.generation_config.top_p = None
            except Exception:
                pass
        generation_pipeline = pipeline(
            "text-generation",
            model=model_ref,
            tokenizer=tokenizer,
            device=pipeline_device,
            max_new_tokens=512,
            do_sample=do_sample_default,
            # If do_sample is False, temperature is ignored but triggers a warning when non-default.
            temperature=(temperature if do_sample_default else 1.0),
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model {hf_model_name}: {e}\nFalling back to 'distilgpt2'.")
        try:
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            model_ref = AutoModelForCausalLM.from_pretrained("distilgpt2")
            generation_pipeline = pipeline(
                "text-generation",
                model=model_ref,
                tokenizer=tokenizer,
                device=pipeline_device if resolved_device != "mps" else -1,
                max_new_tokens=256,
                do_sample=do_sample_default,
                temperature=(temperature if do_sample_default else 1.0),
            )
        except Exception as e2:
            print(f"[FATAL] Could not load fallback model: {e2}. You can set HUGGINGFACE_LLM_MODEL=sshleifer/tiny-gpt2 for a very small test model.")
            generation_pipeline = None
    return generation_pipeline

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_BASE = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")

def sarvam_translate(text: str, target_lang: str = "en") -> str:
    """Best-effort translation to target_lang.

    Priority:
    1) Sarvam API if SARVAM_API_KEY set
    2) deep-translator GoogleTranslator fallback (no API key)
    3) Return input text on failure
    """
    text = text or ""
    if not text.strip():
        return text
    # 1) Sarvam API
    if SARVAM_API_KEY:
        try:
            url = f"{SARVAM_BASE}/v1/translate"
            payload = {"text": text, "target_language": target_lang}
            headers = {"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"}
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            if r.status_code == 200:
                data = r.json()
                return data.get("translated_text", text)
        except Exception:
            pass
    # 2) Fallback: deep-translator
    if _DEEPLITE_AVAILABLE:
        try:
            return GoogleTranslator(source="auto", target=target_lang).translate(text)
        except Exception:
            pass
    # 3) Give up
    return text

def detect_language_simple(text: str) -> str:
    """Return ISO-ish language code if detectable.

    Priority:
    1. langdetect if installed (best-effort; may raise errors on short text)
    2. ASCII heuristic fallback -> 'en' if >95% ascii else 'unknown'
    """
    if _LANGDETECT_AVAILABLE:
        try:
            # langdetect can misclassify very short strings; guard minimal length
            if len(text.strip()) >= 3:
                lang = detect(text)
                # Normalize some frequent outputs to 'en'
                if lang in {"en", "en-us", "en-uk"}:
                    return "en"
                return lang
        except Exception:
            pass
    ascii_ratio = sum(c.isascii() for c in text) / max(1, len(text) or 1)
    return "en" if ascii_ratio > 0.95 else "unknown"


# ------------- Voice to Text (ASR) -------------
try:
    from faster_whisper import WhisperModel  # type: ignore
    _ASR_AVAILABLE = True
except Exception:
    _ASR_AVAILABLE = False


def transcribe_audio(audio_bytes: bytes, language_hint: Optional[str] = None) -> str:
    """Transcribe WAV audio bytes to text using faster-whisper.

    Requirements: pip install faster-whisper and ensure ffmpeg is available on PATH.
    Accepts WAV input (mono/16k recommended). Returns a best-effort transcript.
    """
    if not _ASR_AVAILABLE:
        raise RuntimeError("ASR not available. Install 'faster-whisper' to enable voice input.")
    device_asr = os.getenv("ASR_DEVICE", "cpu")  # cpu or cuda
    compute_type = os.getenv("ASR_COMPUTE_TYPE", ("int8" if device_asr == "cpu" else "float16"))
    model_size = os.getenv("ASR_MODEL", "small")  # tiny, base, small, medium, large-v2
    model = WhisperModel(model_size, device=device_asr, compute_type=compute_type)
    # Write to a temp file for decoding
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        segments, info = model.transcribe(tmp_path, language=language_hint, vad_filter=True)
        text_parts = [seg.text.strip() for seg in segments if getattr(seg, "text", None)]
        return " ".join(text_parts).strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def build_prompt(question: str, docs_text: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Relevant Documents:\n{docs_text}\n\n"
        "Provide a concise factual answer based ONLY on the documents above. If the answer is not present, reply exactly: I'm not sure."
    )

def answer_query(query: str, retriever, messages=None):
    if messages is None:
        messages = [SystemMessage(content="You are a helpful assistant that analyzes health report data.")]
    original_lang = detect_language_simple(query)
    # Translate incoming query to English for retrieval if needed.
    # If language is unknown, still attempt English translation to improve retrieval.
    working_query = query if original_lang == "en" else sarvam_translate(query, target_lang="en")
    if original_lang == "unknown":
        translated = sarvam_translate(query, target_lang="en")
        if translated and translated.strip():
            working_query = translated
    relevant_docs = retriever.invoke(working_query)
    docs_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    combined_prompt = build_prompt(working_query, docs_text)
    # We intentionally allow heterogeneous message objects; type checkers may warn but runtime is fine.
    messages.append(HumanMessage(content=combined_prompt))
    pipe = load_generation_pipeline()
    # If hosted inference is forced but we don't have credentials, bail out early.
    if use_hosted_inference and not _hf_token():
        return {
            "answer": None,
            "answer_en": None,
            "context": docs_text,
            "note": "USE_HF_INFERENCE_API=true but no HF token found. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.",
            "query_original": query,
            "query_en": working_query,
            "query_lang": original_lang,
        }

    # If local pipeline is unavailable, try hosted inference if possible; otherwise return retrieval-only.
    if pipe is None and not _hf_token():
        return {
            "answer": None,
            "answer_en": None,
            "context": docs_text,
            "note": "No generation pipeline available.",
            "query_original": query,
            "query_en": working_query,
            "query_lang": original_lang,
        }
    try:
        # Use hosted inference when explicitly enabled, or when local pipeline is missing.
        if use_hosted_inference or pipe is None:
            response_text = _hf_hosted_generate(combined_prompt, model=hf_model_name, max_new_tokens=512)
            response_text = (response_text or "").strip()
        else:
            raw_output = pipe(
                combined_prompt,
                do_sample=False,
                num_return_sequences=1,
            )[0]["generated_text"]
            # Remove the prompt portion robustly (handle potential whitespace differences)
            if raw_output.startswith(combined_prompt):
                response_text = raw_output[len(combined_prompt):]
            else:
                # Attempt split by prompt sentinel lines
                parts = raw_output.split("Relevant Documents:")
                response_text = raw_output
                if len(parts) > 1:
                    # Take last segment after docs block
                    response_text = parts[-1]
            response_text = response_text.strip()

        # Remove repeated instruction sentence if model echoes it multiple times
        dedupe_line = "If the answer is not present, reply exactly: I'm not sure."
        if response_text:
            lines = [l for l in response_text.splitlines() if l.strip()]
            cleaned = []
            prev = None
            for l in lines:
                if l.strip() == dedupe_line and prev == dedupe_line:
                    continue
                cleaned.append(l)
                prev = l.strip()
            response_text = "\n".join(cleaned)
            unique_non_instr = [l for l in cleaned if l.strip() != dedupe_line]
            if not unique_non_instr:
                response_text = "I'm not sure."
        else:
            response_text = "I'm not sure."
    except Exception as gen_err:
        response_text = f"Generation failed: {gen_err}"
    if len(response_text) > 4000:
        response_text = response_text[:4000]

    # Preserve English answer before translating back to original language for display
    answer_en = response_text
    answer_display = response_text if original_lang == "en" else sarvam_translate(response_text, target_lang=original_lang)

    messages.append(AIMessage(content=answer_display))
    return {
        "answer": answer_display,
        "answer_en": answer_en,
        "context": docs_text,
        "note": None,
        "query_original": query,
        "query_en": working_query,
        "query_lang": original_lang,
        "answer_lang": original_lang,
    }

def interactive_mode(retriever):
    messages = [SystemMessage(content="You are a helpful assistant that analyzes health report data.")]
    print("Entering interactive mode. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        result = answer_query(query, retriever, messages)
        if result["answer"] is None:
            print("[WARN] No model answer. Showing retrieved context snippet:\n")
            print(result["context"][:1500])
        else:
            print("\n--- LLM Response ---")
            print(result["answer"])

def single_shot_mode(retriever, query: str):
    result = answer_query(query, retriever)
    if result["answer"] is None:
        print("[NO_ANSWER] Model unavailable. Retrieved context snippet:\n")
        print(result["context"][:1500])
    else:
        print("--- LLM Response ---")
        print(result["answer"])

def parse_args():
    parser_local = argparse.ArgumentParser(description="Unified RAG QA (interactive or single question)")
    parser_local.add_argument("--query", "-q", type=str, help="Single question to answer (activates single-shot mode)")
    parser_local.add_argument("--mode", choices=["auto", "interactive", "single"], default="auto", help="Mode selection: auto picks interactive if no --query provided")
    parser_local.add_argument("--retrieval", choices=["dense", "hybrid"], default=os.getenv("RETRIEVAL_MODE", "dense"), help="Retrieval strategy: dense or hybrid (dense+bm25)")
    parser_local.add_argument("--k", type=int, default=int(os.getenv("RETRIEVER_K", "3")), help="Top-k chunks to retrieve")
    parser_local.add_argument("--audio", type=str, help="Path to a WAV audio file to transcribe and use as the query")
    return parser_local.parse_args()

def main():
    args = parse_args()
    retriever = init_vector_store(mode=args.retrieval, k=args.k)
    if args.audio and not args.query:
        # Load and transcribe audio input if provided
        try:
            with open(args.audio, "rb") as f:
                audio_bytes = f.read()
            transcript = transcribe_audio(audio_bytes)
            print(f"[INFO] Transcribed query: {transcript}")
            args.query = transcript
        except Exception as e:
            print(f"[ERROR] Failed to transcribe audio: {e}")
            raise SystemExit(1)
    if args.mode == "interactive" or (args.mode == "auto" and not args.query):
        interactive_mode(retriever)
    else:
        if not args.query:
            raise SystemExit("--query is required for single-shot mode (or omit for interactive).")
        single_shot_mode(retriever, args.query)

if __name__ == "__main__":
    main()


