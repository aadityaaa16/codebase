"""
config.py

Single place that decides whether the API uses the REAL embedder/explainer
(sentence-transformers + Gemini) or the SANDBOX MOCKS (TF-IDF + templates).

Why centralize this: every other file just calls `get_embed_fn()` or
`get_explain_fn()` and doesn't need to know or care which implementation
is behind it. This is the same "swap the model without touching the
rest of the system" principle from the chunker/embedder design.

How it decides:
- EMBEDDING_MODE env var: "real" or "mock" (default: "mock")
- EXPLAIN_MODE env var: "real" or "mock" (default: "mock")

On your own machine, before running the app:
    export EMBEDDING_MODE=real
    export EXPLAIN_MODE=real
    export GEMINI_API_KEY=your_key_here
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "indexer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "mock")
EXPLAIN_MODE = os.environ.get("EXPLAIN_MODE", "mock")


def get_embed_fn():
    """
    Returns the embed(texts) -> vectors function to use, based on
    EMBEDDING_MODE. Also returns whether fit() needs to be called first
    (only the mock TF-IDF embedder needs this).
    """
    if EMBEDDING_MODE == "real":
        from embedder import embed
        return embed, False  # needs_fit = False
    else:
        from embedder_mock import embed
        return embed, True  # needs_fit = True


def get_fit_fn():
    """Returns the fit(corpus) function, only relevant for the mock embedder."""
    if EMBEDDING_MODE == "mock":
        from embedder_mock import fit
        return fit
    return None


def get_explain_fn():
    """Returns the explain(question, chunks) -> str function to use."""
    if EXPLAIN_MODE == "real":
        from explainer import explain
        return explain
    else:
        from explainer_mock import explain
        return explain
