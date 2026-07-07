import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "indexer"))
sys.path.append(os.path.dirname(__file__))

from run_indexer import index_repo
import vector_store
from bm25_search import BM25Index


def main():
    chunks = index_repo(os.path.join(os.path.dirname(__file__), "..", "sample_repo"))
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

    index = BM25Index()
    index.build(ids, texts, metadatas)

    test_questions = [
        "create access token",
        "send password reset email",
        "verify password",
    ]

    for question in test_questions:
        print(f'QUESTION: "{question}"')
        results = index.search(question, n_results=3)
        for r in results:
            m = r["metadata"]
            location = f"{m['file_path']}:{m['start_line']}-{m['end_line']}"
            print(f"  [{r['score']:.3f}] {m['chunk_type']:<9} {m['name']:<25} ({location})")
        print()


if __name__ == "__main__":
    main()
