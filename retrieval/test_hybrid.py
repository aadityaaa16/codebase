import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "indexer"))
sys.path.append(os.path.dirname(__file__))

from run_indexer import index_repo
from embedder_mock import fit, embed
import vector_store
from bm25_search import BM25Index
from hybrid_search import hybrid_search


def main():
    repo_path = os.path.join(os.path.dirname(__file__), "..", "sample_repo")
    chunks = index_repo(repo_path)

    texts = [vector_store._chunk_to_text(c) for c in chunks]
    ids = [f"{c.file_path}::{c.name}::{c.start_line}" for c in chunks]
    metadatas = [
        {
            "file_path": c.file_path,
            "chunk_type": c.chunk_type,
            "name": c.name,
            "parent_class": c.parent_class or "",
            "start_line": c.start_line,
            "end_line": c.end_line,
        }
        for c in chunks
    ]

    # Set up semantic search
    fit(texts)
    vector_store.reset_collection()
    vector_store.store_chunks(chunks, embed)

    # Set up BM25
    bm25_index = BM25Index()
    bm25_index.build(ids, texts, metadatas)

    test_questions = ["create access token", "send password reset email", "verify password"]

    for question in test_questions:
        print(f'QUESTION: "{question}"')
        print("  HYBRID RESULTS:")
        results = hybrid_search(question, embed, bm25_index, alpha=0.5, n_results=3)
        for r in results:
            m = r["metadata"]
            location = f"{m['file_path'].split('sample_repo/')[-1]}:{m['start_line']}-{m['end_line']}"
            print(f"    [{r['hybrid_score']:.3f}] (sem={r['semantic_norm']:.2f} bm25={r['bm25_norm']:.2f}) "
                  f"{m['chunk_type']:<9} {m['name']:<25} ({location})")
        print()


if __name__ == "__main__":
    main()
