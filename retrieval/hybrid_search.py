"""
hybrid_search.py

Combines semantic (vector/embedding) search with BM25 keyword search
into a single ranked list.

Why we can't just merge them directly:
- ChromaDB gives a DISTANCE (lower = more similar)
- BM25 gives a SCORE (higher = more relevant)
These are on different scales and point in opposite directions, so we
normalize both onto a common 0-1 scale (1 = best) before combining.

The combination is a simple weighted average:
    hybrid_score = alpha * semantic_score + (1 - alpha) * bm25_score

alpha=0.5 means "trust both equally" - a reasonable default. Raising
alpha favors semantic/conceptual matches; lowering it favors exact
keyword/identifier matches. We expose it as a parameter rather than
hardcoding it, since this is a genuine tunable a recruiter might ask
about ("how did you decide the weighting?").
"""

from typing import List, Dict
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "indexer"))
sys.path.append(os.path.dirname(__file__))

import vector_store
from bm25_search import BM25Index


def _normalize(scores: List[float], reverse: bool = False) -> List[float]:
    """
    Min-max normalizes a list of scores to the 0-1 range.
    reverse=True is used for distances, where LOWER is better, so we
    flip the scale so that 1.0 always means "best" regardless of
    whether the original metric was a distance or a score.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]  # all equal - avoid divide by zero
    normalized = [(s - lo) / (hi - lo) for s in scores]
    if reverse:
        normalized = [1.0 - n for n in normalized]
    return normalized


def hybrid_search(
    question: str,
    embed_fn,
    bm25_index: BM25Index,
    alpha: float = 0.5,
    n_results: int = 5,
    candidate_pool: int = 15,
) -> List[Dict]:
    """
    Runs both semantic and BM25 search, normalizes and combines their
    scores, and returns the top n_results overall.

    candidate_pool: how many results to pull from EACH method before
    combining. We pull more than n_results because a chunk might rank
    #12 in one method but #1 in the other - if we only pulled the top 5
    from each, we might miss it entirely from the combined ranking.
    """
    semantic_results = vector_store.query(question, embed_fn, n_results=candidate_pool)
    bm25_results = bm25_index.search(question, n_results=candidate_pool)

    # Build score maps keyed by chunk id, so we can merge by identity
    semantic_scores = {r["id"]: r["distance"] for r in semantic_results}
    bm25_scores = {r["id"]: r["score"] for r in bm25_results}

    # Union of all chunk ids seen by either method
    all_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())

    # We need a lookup for text/metadata regardless of which method found it
    info_by_id = {}
    for r in semantic_results:
        info_by_id[r["id"]] = {"text": r["text"], "metadata": r["metadata"]}
    for r in bm25_results:
        info_by_id.setdefault(r["id"], {"text": r["text"], "metadata": r["metadata"]})

    # Normalize: for chunks missing from one method entirely, treat them
    # as the worst possible score for that method (0.0), rather than
    # skipping - a chunk that BOTH methods ignored shouldn't rank highly.
    ordered_ids = list(all_ids)

    raw_semantic = [semantic_scores.get(i, None) for i in ordered_ids]
    raw_bm25 = [bm25_scores.get(i, None) for i in ordered_ids]

    present_semantic_vals = [v for v in raw_semantic if v is not None]
    present_bm25_vals = [v for v in raw_bm25 if v is not None]

    norm_semantic_vals = _normalize(present_semantic_vals, reverse=True)  # distance: lower=better
    norm_bm25_vals = _normalize(present_bm25_vals, reverse=False)         # score: higher=better

    # Map normalized values back to their ids
    sem_iter = iter(norm_semantic_vals)
    bm25_iter = iter(norm_bm25_vals)
    norm_semantic = {}
    norm_bm25 = {}
    for i, v in zip(ordered_ids, raw_semantic):
        norm_semantic[i] = next(sem_iter) if v is not None else 0.0
    for i, v in zip(ordered_ids, raw_bm25):
        norm_bm25[i] = next(bm25_iter) if v is not None else 0.0

    combined = []
    for chunk_id in ordered_ids:
        hybrid_score = alpha * norm_semantic[chunk_id] + (1 - alpha) * norm_bm25[chunk_id]
        combined.append({
            "id": chunk_id,
            "hybrid_score": hybrid_score,
            "semantic_norm": norm_semantic[chunk_id],
            "bm25_norm": norm_bm25[chunk_id],
            "text": info_by_id[chunk_id]["text"],
            "metadata": info_by_id[chunk_id]["metadata"],
        })

    combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return combined[:n_results]
