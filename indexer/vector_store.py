"""
vector_store.py

Wraps ChromaDB so the rest of the project doesn't need to know
ChromaDB's API directly - just `store_chunks()` and `query()`.

Each chunk is stored with:
- its embedding vector (for similarity search)
- its actual code text (so we can show/explain it later)
- metadata: file_path, chunk_type, name, parent_class, start_line, end_line
  (this is what lets us answer "Answer: auth/jwt.py, lines 8-16" instead
  of just returning raw text with no source)
"""

import chromadb
from typing import List
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from chunker import CodeChunk

_client = None
_collection = None

COLLECTION_NAME = "code_chunks"

# IMPORTANT: this lives OUTSIDE the project folder (in the user's home
# directory), not inside it. If ChromaDB wrote its files inside the
# project directory, `uvicorn --reload` would detect those new/changed
# files as source code changes and restart the server mid-request -
# which produces exactly the "empty response body" bug this was fixed
# to avoid.
_DATA_DIR = Path.home() / ".codebase_navigator" / "chroma_db"


def _get_collection():
    global _client, _collection
    if _client is None:
        # Persistent client so the index survives between runs -
        # you don't want to re-embed the whole repo every time you ask a question
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(_DATA_DIR))
    if _collection is None:
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def reset_collection():
    """
    Wipes the existing collection. Useful when re-indexing from scratch
    during development, so old/stale chunks don't linger.
    """
    global _collection
    client = _get_collection().name  # ensure client initialized
    _client.delete_collection(COLLECTION_NAME)
    _collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def _chunk_to_text(chunk: CodeChunk) -> str:
    """
    Builds the actual text we embed. We include the docstring and
    file path as context, not just raw code, because a question phrased
    in plain English matches docstring language better than raw syntax.
    """
    parts = []
    if chunk.docstring:
        parts.append(chunk.docstring)
    parts.append(chunk.code)
    return "\n".join(parts)


def store_chunks(chunks: List[CodeChunk], embed_fn):
    """
    Embeds and stores a list of CodeChunks.
    `embed_fn` is injected (rather than imported directly) so this file
    doesn't care whether it's the real Gemini embedder
    or the sandbox mock - same pattern as the rest of the project.
    """
    collection = _get_collection()

    texts = [_chunk_to_text(c) for c in chunks]
    vectors = embed_fn(texts)

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

    collection.add(
        ids=ids,
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def query(question: str, embed_fn, n_results: int = 5):
    """
    Embeds the question and returns the n_results most similar chunks,
    each with its source file, line numbers, and the matched text.
    """
    collection = _get_collection()
    question_vector = embed_fn([question])

    results = collection.query(
        query_embeddings=question_vector.tolist(),
        n_results=n_results,
    )

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return matches
