"""
embedder.py

PRODUCTION embedder - uses Gemini's embedding API (gemini-embedding-001)
instead of a local sentence-transformers model.

Why this version: it uses the SAME free Gemini API key you already have
for the explanation layer, avoids installing PyTorch (a large, heavy
dependency), and keeps the whole project's install lightweight.

Trade-off vs. the local sentence-transformers approach: this requires
an internet connection and an API key, and is subject to Gemini's free
tier rate limits - both fine for a portfolio project's normal usage.

Set GEMINI_API_KEY as an environment variable before running:
    export GEMINI_API_KEY=your_key_here
Get a free key at: https://aistudio.google.com/apikey
"""

import os
from google import genai
from typing import List
import numpy as np

_client = None

EMBEDDING_MODEL = "gemini-embedding-001"


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Get a free key at https://aistudio.google.com/apikey "
                "and set it before running: export GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def embed(texts: List[str]) -> np.ndarray:
    """
    Converts a list of text strings (e.g. code chunks) into embedding
    vectors using Gemini's embedding API.

    Note: gemini-embedding-001 processes one text per request, so we
    loop rather than send a single batched call. For a small portfolio
    project (tens to low hundreds of chunks) this is fast enough; a
    much larger repo would want to parallelize or batch these calls.
    """
    client = _get_client()
    vectors = []
    for text in texts:
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        vectors.append(result.embeddings[0].values)
    return np.array(vectors)


if __name__ == "__main__":
    # Quick sanity check when run directly on your own machine
    sample = [
        "def create_access_token(user_id): return jwt.encode(payload, SECRET_KEY)",
        "def send_password_reset_email(email, token): print('sending email')",
    ]
    vectors = embed(sample)
    print("Embedding shape:", vectors.shape)
    print("First 5 dims of chunk 1:", vectors[0][:5])
