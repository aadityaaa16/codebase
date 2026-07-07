"""
bm25_search.py

BM25 keyword search over code chunks. Unlike embeddings (which capture
meaning), BM25 scores chunks based on exact term overlap with the query -
weighted by how rare/important each term is across the whole corpus.

Why we need this alongside embeddings:
A question like "where is create_access_token defined?" contains a very
specific identifier. Embeddings might get distracted by other
token/auth-related chunks. BM25 directly rewards the chunk that
literally contains the term "create_access_token", which is exactly
what we want for identifier lookups.
"""

import re
from rank_bm25 import BM25Okapi
from typing import List, Dict
import sys
import os

sys.path.append(os.path.dirname(__file__))


def _tokenize(text: str) -> List[str]:
    """
    Splits code into searchable tokens. This is more than a plain
    `.split()` because code identifiers use snake_case and camelCase -
    we want "create_access_token" to also be searchable as
    "create", "access", "token" individually, since a user might type
    the concept in plain English rather than the exact identifier.
    """
    # Split snake_case
    text = text.replace("_", " ")
    # Split camelCase (insert space before capital letters)
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", text)
    # Lowercase and extract word tokens
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """
    Wraps rank_bm25 with our tokenization and keeps track of which
    chunk each score corresponds to.
    """

    def __init__(self):
        self.bm25 = None
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []
        self.chunk_metadata: List[Dict] = []

    def build(self, ids: List[str], texts: List[str], metadatas: List[Dict]):
        """
        Builds the BM25 index over the given chunks. Must be called
        once after indexing, before search() can be used.
        """
        self.chunk_ids = ids
        self.chunk_texts = texts
        self.chunk_metadata = metadatas

        tokenized_corpus = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Returns the top n_results chunks ranked by BM25 score
        (higher score = better match, unlike ChromaDB's distance
        where lower = better - worth remembering when we combine them).
        """
        if self.bm25 is None:
            raise RuntimeError("Call build() before search().")

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair scores with their chunk info, sort descending
        ranked = sorted(
            zip(scores, self.chunk_ids, self.chunk_texts, self.chunk_metadata),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk_id, text, metadata in ranked[:n_results]:
            results.append({
                "id": chunk_id,
                "score": float(score),
                "text": text,
                "metadata": metadata,
            })
        return results
