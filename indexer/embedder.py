"""
embedder.py

PRODUCTION embedder. Run this on your own machine (not in a network-
restricted sandbox), since it downloads a small model from Hugging Face
the first time it runs (~80MB, cached afterward).

Uses `sentence-transformers` with `all-MiniLM-L6-v2`:
- Free, no API key required
- Runs entirely on CPU, fine for an 8GB RAM laptop
- Produces a 384-dimensional vector per chunk of text

Any module in this project that needs embeddings should import
`embed()` from here (or from embedder_mock.py during sandbox testing)
so the rest of the code never has to know which one is in use.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

_model = None


def _get_model() -> SentenceTransformer:
    """
    Lazily loads the model once and reuses it across calls, since
    loading it is the slow part (a few seconds), not the encoding itself.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """
    Converts a list of text strings (e.g. code chunks) into embedding
    vectors. Returns a numpy array of shape (len(texts), 384).
    """
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


if __name__ == "__main__":
    # Quick sanity check when run directly on your own machine
    sample = [
        "def create_access_token(user_id): return jwt.encode(payload, SECRET_KEY)",
        "def send_password_reset_email(email, token): print('sending email')",
    ]
    vectors = embed(sample)
    print("Embedding shape:", vectors.shape)
    print("First 5 dims of chunk 1:", vectors[0][:5])
