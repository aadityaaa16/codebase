"""
main.py

FastAPI backend for the AI Codebase Navigator.

Two endpoints:
- POST /index  -> point it at a repo path, it chunks + embeds + stores everything
- POST /query  -> ask a question, get back a grounded explanation + source files

Run locally with:
    uvicorn main:app --reload

Then visit http://localhost:8000/docs for interactive API docs (FastAPI
generates this automatically - worth mentioning in interviews, it's a
big reason FastAPI is a strong choice over Flask for this kind of project).
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "indexer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from run_indexer import index_repo
import vector_store
from bm25_search import BM25Index
from hybrid_search import hybrid_search
import config

app = FastAPI(title="AI Codebase Navigator")

# In-memory state. Fine for a single-user portfolio project - a real
# production version would persist this per-project rather than in
# process memory, but that's out of scope for v1.
_state = {
    "bm25_index": None,
    "indexed_repo": None,
    "chunk_count": 0,
}


class IndexRequest(BaseModel):
    repo_path: str


class IndexResponse(BaseModel):
    repo_path: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    alpha: float = 0.5  # semantic vs BM25 weighting, exposed so the UI can let users tune it


class SourceReference(BaseModel):
    file_path: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    hybrid_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceReference]


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "indexed_repo": _state["indexed_repo"],
        "chunk_count": _state["chunk_count"],
        "embedding_mode": config.EMBEDDING_MODE,
        "explain_mode": config.EXPLAIN_MODE,
    }


@app.post("/index", response_model=IndexResponse)
def index_repository(req: IndexRequest):
    """
    Chunks every Python file in the given repo path, embeds each chunk,
    and stores them in ChromaDB + builds the BM25 index. Must be called
    once before /query will return anything useful.
    """
    if not os.path.isdir(req.repo_path):
        raise HTTPException(status_code=400, detail=f"Path not found: {req.repo_path}")

    chunks = index_repo(req.repo_path)
    if not chunks:
        raise HTTPException(status_code=400, detail="No Python files with functions/classes found in this path.")

    embed_fn, needs_fit = config.get_embed_fn()

    texts = [vector_store._chunk_to_text(c) for c in chunks]
    ids = [f"{c.file_path}::{c.name}::{c.start_line}" for c in chunks]
    metadatas = [
        {
            "file_path": c.file_path, "chunk_type": c.chunk_type, "name": c.name,
            "parent_class": c.parent_class or "", "start_line": c.start_line, "end_line": c.end_line,
        }
        for c in chunks
    ]

    if needs_fit:
        fit_fn = config.get_fit_fn()
        fit_fn(texts)

    vector_store.reset_collection()
    vector_store.store_chunks(chunks, embed_fn)

    bm25_index = BM25Index()
    bm25_index.build(ids, texts, metadatas)
    _state["bm25_index"] = bm25_index
    _state["indexed_repo"] = req.repo_path
    _state["chunk_count"] = len(chunks)

    return IndexResponse(repo_path=req.repo_path, chunks_indexed=len(chunks))


@app.post("/query", response_model=QueryResponse)
def query_codebase(req: QueryRequest):
    """
    Runs hybrid search against the indexed repo and returns a grounded
    explanation plus the source files/lines the answer came from.
    """
    if _state["bm25_index"] is None:
        raise HTTPException(status_code=400, detail="No repository indexed yet. Call POST /index first.")

    embed_fn, _ = config.get_embed_fn()
    explain_fn = config.get_explain_fn()

    results = hybrid_search(
        req.question, embed_fn, _state["bm25_index"],
        alpha=req.alpha, n_results=req.n_results,
    )

    answer = explain_fn(req.question, results)

    sources = [
        SourceReference(
            file_path=r["metadata"]["file_path"],
            chunk_type=r["metadata"]["chunk_type"],
            name=r["metadata"]["name"],
            start_line=r["metadata"]["start_line"],
            end_line=r["metadata"]["end_line"],
            hybrid_score=r["hybrid_score"],
        )
        for r in results
    ]

    return QueryResponse(question=req.question, answer=answer, sources=sources)
