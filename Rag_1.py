import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import pandas as pd
from langchain_chroma import Chroma
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

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "train.parquet")  # Parquet dataset
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

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Load parquet dataset instead of treating it as raw text
    print("\n--- Loading parquet dataset ---")
    if not file_path.lower().endswith('.parquet'):
        raise ValueError("Expected a parquet file. Update path or logic for other formats.")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet file {file_path}: {e}")

    # Optional: limit rows for faster local builds
    max_rows_env = os.getenv("INGEST_MAX_ROWS", "0").strip()
    try:
        max_rows = int(max_rows_env) if max_rows_env else 0
    except ValueError:
        max_rows = 0
    if max_rows > 0 and len(df) > max_rows:
        print(f"[INFO] Limiting ingestion to first {max_rows} rows (of {len(df)}). Set INGEST_MAX_ROWS=0 to disable.")
        df = df.head(max_rows)

    # Decide which columns to merge into textual content.
    # Heuristic: include object (string) columns + selected numeric columns converted with labels.
    text_columns = [c for c in df.columns if df[c].dtype == 'object']
    if not text_columns:
        # Fallback: take all columns
        text_columns = list(df.columns)

    # Build documents list (faster than iterrows)
    documents = []
    for row in df.itertuples(index=True, name=None):
        idx = row[0]
        parts = []
        for i, col in enumerate(text_columns, start=1):
            try:
                val = getattr(row, col)
            except Exception:
                # Fallback if attribute access fails (e.g., non-identifier column names)
                val = df.loc[idx, col]
            if pd.isna(val):
                continue
            parts.append(f"{col}: {val}")
        if not parts:
            continue
        content = "\n".join(parts)
        documents.append(Document(page_content=content, metadata={"row_index": int(idx)}))

    if not documents:
        raise ValueError("No documents were generated from the parquet file. Check column types or adjust logic.")

    print(f"Loaded {len(documents)} rows -> {len(documents)} raw documents")

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

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
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists and looks complete. No need to initialize.")
