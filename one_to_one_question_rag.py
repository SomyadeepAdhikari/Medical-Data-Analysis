"""
Thin wrapper for single-question RAG using the unified implementation in Rag_final.py.

This script preserves the old entry point while delegating all logic to Rag_final,
so there's a single source of truth for retrieval, translation, and generation.
"""

import os
from dotenv import load_dotenv
from Rag_final import init_vector_store, single_shot_mode


def main():
    load_dotenv()
    query = os.getenv(
        "SAMPLE_QUERY",
        "What are the main findings about Coronary Artery Disease in the dataset?",
    )

    retriever = init_vector_store()
    single_shot_mode(retriever, query)


if __name__ == "__main__":
    main()



