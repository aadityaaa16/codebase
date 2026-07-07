"""
embedder_mock.py

SANDBOX-ONLY stand-in for embedder.py. This exists purely because this
particular development sandbox cannot reach huggingface.co to download
the real sentence-transformers model.

It exposes the SAME `embed(texts) -> vectors` interface as embedder.py,
so every other module (indexing, retrieval) works identically regardless
of which one is plugged in. This lets us test the full pipeline here
and then swap in the real embedder with a one-line import change when
you run this on your own machine.

IMPORTANT: TF-IDF vectors are NOT true semantic embeddings. They match
based on shared words/tokens, not meaning. Do not use this for your
actual project - swap to embedder.py before demoing or submitting.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np

_vectorizer = None
_fitted = False


def fit(corpus: List[str]):
    """
    TF-IDF needs to see the whole corpus once to build its vocabulary,
    unlike a real embedding model which works on any single text
    independently. We call this once during indexing.
    """
    global _vectorizer, _fitted
    _vectorizer = TfidfVectorizer(max_features=384)  # match real embedder's dim for consistency
    _vectorizer.fit(corpus)
    _fitted = True


def embed(texts: List[str]) -> np.ndarray:
    """
    Converts texts into TF-IDF vectors. Must call fit() first with
    the full corpus at least once.
    """
    if not _fitted:
        raise RuntimeError("Call fit(corpus) once before embed() when using the mock embedder.")
    vectors = _vectorizer.transform(texts).toarray()
    return vectors


if __name__ == "__main__":
    sample = [
        "def create_access_token(user_id): return jwt.encode(payload, SECRET_KEY)",
        "def send_password_reset_email(email, token): print('sending email')",
    ]
    fit(sample)
    vectors = embed(sample)
    print("Embedding shape:", vectors.shape)
