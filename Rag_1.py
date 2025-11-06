import os
import glob
from pathlib import Path

# --- NumPy compatibility shims for NumPy 2.x (used by some deps like chromadb) ---
try:
    import numpy as _np  # type: ignore
    if not hasattr(_np, "float_"):
        _np.float_ = _np.float64  # type: ignore[attr-defined]
    if not hasattr(_np, "uint"):
        _np.uint = _np.uint32  # type: ignore[attr-defined]
except Exception:
    pass
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
from langchain_chroma import Chroma
from chromadb.config import Settings  # disable telemetry
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    # Fast, torch-free embedding backend
    from langchain_community.embeddings import FastEmbedEmbeddings  # type: ignore
    _FASTEMBED = True
except Exception:
    _FASTEMBED = False

# NOTE: Replaced GoogleGenerativeAIEmbeddings with a free HuggingFace embedding model.
# Default model: "sentence-transformers/all-MiniLM-L6-v2" (small, fast, multilingual-ish coverage)
# You can change via environment variable HUGGINGFACE_EMBEDDING_MODEL.

# Define the directory containing source data and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "Documents")  # Capitalized to match repo
parquet_path = os.path.join(data_dir, "train.parquet")  # Preferred if available
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists or needs rebuild
rebuild_env = os.getenv("REBUILD_DB", "false").lower() in {"1", "true", "yes"}
needs_build = False
if not os.path.exists(persistent_directory):
    needs_build = True
else:
    try:
        entries = os.listdir(persistent_directory)
        has_sqlite = os.path.exists(os.path.join(persistent_directory, "chroma.sqlite3"))
        needs_build = (len(entries) == 0) or (not has_sqlite)
    except Exception:
        needs_build = True

if rebuild_env or needs_build:
    if rebuild_env:
        print("REBUILD_DB=true -> Forcing vector store rebuild...")
    elif needs_build:
        print("Vector store missing or incomplete. Initializing vector store...")

    # Load data from parquet if present, else fall back to txt/md files
    df = None
    documents = []
    if os.path.exists(parquet_path):
        print("\n--- Loading parquet dataset ---")
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet file {parquet_path}: {e}")
    else:
        print("\n--- Parquet not found; falling back to text files in Documents/ ---")
        txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        md_files = sorted(glob.glob(os.path.join(data_dir, "*.md")))
        all_files = txt_files + md_files
        if not all_files:
            raise FileNotFoundError(
                "No data files found. Provide Documents/train.parquet or one/more .txt/.md files in Documents/."
            )
        for fp in all_files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    documents.append(Document(page_content=content, metadata={"source": Path(fp).name}))
            except Exception as e:
                print(f"[WARN] Skipping {fp}: {e}")

    # If dataframe is present, convert rows to Documents
    if df is not None:
        # Optional: limit rows for faster local builds
        # Prefer BUILD_ROW_LIMIT if set, else INGEST_MAX_ROWS; default 0 (no limit)
        max_rows_env = (os.getenv("BUILD_ROW_LIMIT") or os.getenv("INGEST_MAX_ROWS") or "0").strip()
        try:
            max_rows = int(max_rows_env) if max_rows_env else 0
        except ValueError:
            max_rows = 0
        if max_rows > 0 and len(df) > max_rows:
            print(f"[INFO] Limiting ingestion to first {max_rows} rows (of {len(df)}). Set INGEST_MAX_ROWS=0 to disable.")
            df = df.head(max_rows)

        # Decide which columns to merge into textual content.
        text_columns = [c for c in df.columns if df[c].dtype == 'object']
        if not text_columns:
            text_columns = list(df.columns)

        # Build documents list (faster than iterrows)
        tmp_docs = []
        for row in df.itertuples(index=True, name=None):
            idx = row[0]
            parts = []
            for col in text_columns:
                try:
                    val = getattr(row, col)
                except Exception:
                    val = df.loc[idx, col]
                try:
                    if pd.isna(val):
                        continue
                except Exception:
                    pass
                parts.append(f"{col}: {val}")
            if not parts:
                continue
            content = "\n".join(parts)
            tmp_docs.append(Document(page_content=content, metadata={"row_index": int(idx)}))
        if not tmp_docs and not documents:
            raise ValueError("No documents were generated from the parquet file. Check column types or adjust logic.")
        documents.extend(tmp_docs)

    print(f"Loaded {len(documents)} rows -> {len(documents)} raw documents")

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Optional cap chunk count; default 0 (no limit)
    build_chunk_limit_env = (os.getenv("BUILD_CHUNK_LIMIT") or "0").strip()
    try:
        build_chunk_limit = int(build_chunk_limit_env) if build_chunk_limit_env else 0
    except ValueError:
        build_chunk_limit = 100
    if build_chunk_limit > 0 and len(docs) > build_chunk_limit:
        print(f"[INFO] Limiting chunk ingestion to first {build_chunk_limit} chunks (of {len(docs)}). Override with BUILD_CHUNK_LIMIT.")
        docs = docs[:build_chunk_limit]

    # Create embeddings (prefer FastEmbed if available to avoid torch issues)
    embedding_model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = None
    if _FASTEMBED:
        try:
            print("\n--- Creating embeddings (FastEmbed) ---")
            embeddings = FastEmbedEmbeddings()
            print("\n--- Finished creating embeddings with FastEmbed ---")
        except Exception as fe:
            print(f"[WARN] FastEmbed failed: {fe}. Falling back to HuggingFaceEmbeddings: {embedding_model_name}")
    if embeddings is None:
        print("\n--- Creating embeddings (HuggingFace) ---")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print(f"\n--- Finished creating embeddings with model: {embedding_model_name} ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    try:
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persistent_directory,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
            collection_name=os.getenv("CHROMA_COLLECTION", "medical_docs"),
        )
        print("\n--- Finished creating vector store ---")
    except Exception as e:
        print(f"[ERROR] Failed creating vector store: {e}")
        raise

else:
    print("Vector store already exists and looks complete. No need to initialize.")
