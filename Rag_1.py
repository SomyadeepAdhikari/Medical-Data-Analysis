import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# NOTE: Replaced GoogleGenerativeAIEmbeddings with a free HuggingFace embedding model.
# Default model: "sentence-transformers/all-MiniLM-L6-v2" (small, fast, multilingual-ish coverage)
# You can change via environment variable HUGGINGFACE_EMBEDDING_MODEL.

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "train.parquet")  # Parquet dataset
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

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

    # Decide which columns to merge into textual content.
    # Heuristic: include object (string) columns + selected numeric columns converted with labels.
    text_columns = [c for c in df.columns if df[c].dtype == 'object']
    if not text_columns:
        # Fallback: take all columns
        text_columns = list(df.columns)

    # Build documents list
    documents = []
    for idx, row in df.iterrows():
        # Concatenate chosen columns into one textual chunk
        parts = []
        for col in text_columns:
            val = row[col]
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

    # Create embeddings (HuggingFace local / hub model)
    print("\n--- Creating embeddings (HuggingFace) ---")
    embedding_model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"\n--- Finished creating embeddings with model: {embedding_model_name} ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
