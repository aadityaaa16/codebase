"""
explainer_groq.py

Alternative explanation layer using Groq instead of Gemini. Same
`explain(question, chunks) -> str` interface as explainer.py, so
swapping between them is a one-line config change, not a rewrite.

Why Groq as an option: no credit card required at all, a generous
free tier (tens of requests/minute, roughly 1,000+/day depending on
model), and it runs on custom LPU hardware that responds fast. Useful
if Gemini's free tier is being flaky (as it was for this project's
"limit: 0" quota issue) or you'd simply rather not deal with Google
Cloud billing/verification at all.

Get a free key at: https://console.groq.com
Set it as an environment variable before running:
    export GROQ_API_KEY=your_key_here
"""

import os
from groq import Groq
from typing import List, Dict

_client = None

# llama-3.3-70b-versatile is a strong, fast, free-tier-friendly choice.
# Groq's available free models change periodically - check
# https://console.groq.com/docs/models if this model is ever retired.
MODEL = "llama-3.3-70b-versatile"


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set. "
                "Get a free key at https://console.groq.com "
                "and set it before running: export GROQ_API_KEY=your_key_here"
            )
        _client = Groq(api_key=api_key)
    return _client


def _build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Same prompt-building logic as explainer.py - only using ONLY the
    retrieved chunks, citing file/line, and saying so explicitly if the
    chunks don't fully answer the question, rather than guessing.
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
    Sends the question and retrieved chunks to Groq (Llama 3.3 70B)
    and returns a plain-English explanation grounded in the actual code.
    """
    client = _get_client()
    prompt = _build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Quick manual test - requires GROQ_API_KEY to be set
    fake_chunks = [{
        "text": "def create_access_token(user_id: int) -> str:\n    return jwt.encode(...)",
        "metadata": {"file_path": "auth/jwt.py", "start_line": 8, "end_line": 16},
    }]
    print(explain("Where is JWT authentication implemented?", fake_chunks))
