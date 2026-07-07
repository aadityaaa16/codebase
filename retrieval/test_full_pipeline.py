import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "indexer"))
sys.path.append(os.path.dirname(__file__))

from run_indexer import index_repo
from embedder_mock import fit, embed
import vector_store
from bm25_search import BM25Index
from hybrid_search import hybrid_search
from explainer_mock import explain


def ask(question: str, embed_fn, bm25_index: BM25Index):
    print(f'\n{"="*70}\nQUESTION: "{question}"\n{"="*70}')
    chunks = hybrid_search(question, embed_fn, bm25_index, alpha=0.5, n_results=3)
    answer = explain(question, chunks)
    print(answer)


def main():
    repo_path = os.path.join(os.path.dirname(__file__), "..", "sample_repo")
    chunks = index_repo(repo_path)

    texts = [vector_store._chunk_to_text(c) for c in chunks]
    ids = [f"{c.file_path}::{c.name}::{c.start_line}" for c in chunks]
    metadatas = [
        {
            "file_path": c.file_path, "chunk_type": c.chunk_type, "name": c.name,
            "parent_class": c.parent_class or "", "start_line": c.start_line, "end_line": c.end_line,
        }
        for c in chunks
    ]

    fit(texts)
    vector_store.reset_collection()
    vector_store.store_chunks(chunks, embed)

    bm25_index = BM25Index()
    bm25_index.build(ids, texts, metadatas)

    # These are the exact example questions from the original project doc
    ask("Where is JWT authentication implemented?", embed, bm25_index)
    ask("How is password reset implemented?", embed, bm25_index)
    ask("Which endpoint creates a user?", embed, bm25_index)


if __name__ == "__main__":
    main()
