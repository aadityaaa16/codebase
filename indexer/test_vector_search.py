"""
test_vector_search.py

End-to-end test: chunk the sample repo, embed with the SANDBOX MOCK
embedder (TF-IDF), store in ChromaDB, and run a query to see semantic
search working for the first time.

NOTE: because this uses the mock TF-IDF embedder (see embedder_mock.py),
matches here are word-overlap based, not true semantic similarity.
On your own machine, swap `embedder_mock` for `embedder` and matches
will work even when the question uses different words than the code.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from run_indexer import index_repo
from embedder_mock import fit, embed
import vector_store


def main():
    print("Step 1: Chunking sample repo...")
    chunks = index_repo(os.path.join(os.path.dirname(__file__), "..", "sample_repo"))
    print(f"  -> {len(chunks)} chunks found\n")

    print("Step 2: Fitting mock embedder on the full corpus...")
    texts = [vector_store._chunk_to_text(c) for c in chunks]
    fit(texts)
    print("  -> done\n")

    print("Step 3: Resetting and storing chunks in ChromaDB...")
    vector_store.reset_collection()
    count = vector_store.store_chunks(chunks, embed)
    print(f"  -> stored {count} chunks\n")

    print("Step 4: Running test queries...\n")
    test_questions = [
        "create access token",
        "send password reset email",
        "verify password",
    ]

    for question in test_questions:
        print(f"QUESTION: \"{question}\"")
        results = vector_store.query(question, embed, n_results=3)
        for r in results:
            m = r["metadata"]
            location = f"{m['file_path']}:{m['start_line']}-{m['end_line']}"
            print(f"  [{r['distance']:.3f}] {m['chunk_type']:<9} {m['name']:<25} ({location})")
        print()


if __name__ == "__main__":
    main()
