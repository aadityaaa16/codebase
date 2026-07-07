"""
explainer.py

PRODUCTION explanation layer. Takes the user's question plus the
retrieved code chunks (from hybrid_search) and asks Gemini to explain
how the code answers the question - grounded in the actual retrieved
text, with file paths and line numbers.

Run this on your own machine with a real Gemini API key set as an
environment variable: GEMINI_API_KEY

Get a free key at: https://aistudio.google.com/apikey
"""

import os
from google import genai
from typing import List, Dict

_client = None


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


def _build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Builds the prompt sent to Gemini. Key design choice: we explicitly
    instruct the model to ONLY use the provided chunks and to cite file
    paths/line numbers - this is what prevents the model from making up
    plausible-sounding but wrong answers about code it hasn't actually
    seen.
    """
    context_blocks = []
    for c in chunks:
        m = c["metadata"]
        location = f"{m['file_path']} (lines {m['start_line']}-{m['end_line']})"
        context_blocks.append(f"### {location}\n```python\n{c['text']}\n```")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a codebase assistant. Answer the developer's question using ONLY the code chunks provided below. Do not invent details about code you cannot see.

Question: {question}

Retrieved code chunks:
{context}

Instructions:
- Explain how the code answers the question, in plain English.
- Reference specific file paths and line numbers from the chunks above.
- If the provided chunks don't fully answer the question, say so explicitly rather than guessing.
- Keep the explanation concise (3-6 sentences).
"""
    return prompt


def explain(question: str, chunks: List[Dict]) -> str:
    """
    Sends the question and retrieved chunks to Gemini and returns
    a plain-English explanation grounded in the actual code.
    """
    client = _get_client()
    prompt = _build_prompt(question, chunks)

    response = client.models.generate_content(
        model="gemini-2.0-flash",  # fast + free-tier friendly
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    # Quick manual test - requires GEMINI_API_KEY to be set
    fake_chunks = [{
        "text": "def create_access_token(user_id: int) -> str:\n    return jwt.encode(...)",
        "metadata": {"file_path": "auth/jwt.py", "start_line": 8, "end_line": 16},
    }]
    print(explain("Where is JWT authentication implemented?", fake_chunks))
