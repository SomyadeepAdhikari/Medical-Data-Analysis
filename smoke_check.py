"""
Quick smoke check to validate environment and vector store.
Runs in retrieval-only mode (no torch required).
"""
import os
from dotenv import load_dotenv

# Force retrieval-only mode for smoke test
os.environ.setdefault("SKIP_LLM", "true")

from Rag_final import init_vector_store, answer_query  # noqa: E402


def main():
    load_dotenv()
    retriever = init_vector_store(mode=os.getenv("RETRIEVAL_MODE", "dense"), k=int(os.getenv("RETRIEVER_K", "3")))
    result = answer_query("What is in the dataset?", retriever)
    ctx = result.get("context", "")
    print("Context chars:", len(ctx))
    print("Answer:", result.get("note") or result.get("answer") or "<no answer>")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
