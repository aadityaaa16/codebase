"""
explainer_mock.py

SANDBOX-ONLY stand-in for explainer.py. No real LLM call - just formats
the retrieved chunks into a templated, readable answer so we can verify
the full pipeline (question -> retrieval -> "explanation" -> output)
works end-to-end without needing network access to Gemini.

Swap to explainer.py (real Gemini) when running on your own machine.
"""

from typing import List, Dict


def explain(question: str, chunks: List[Dict]) -> str:
    """
    Produces a templated (non-LLM) explanation, purely to prove the
    retrieved chunks are being passed through correctly. This is NOT
    a real explanation - it's a structural placeholder.
    """
    if not chunks:
        return f'No relevant code found for: "{question}"'

    lines = [f'[MOCK EXPLANATION - no real LLM call]\n"{question}" appears related to:\n']
    for c in chunks:
        m = c["metadata"]
        location = f"{m['file_path']} (lines {m['start_line']}-{m['end_line']})"
        chunk_label = f"{m['chunk_type']} `{m['name']}`"
        lines.append(f"- {chunk_label} in {location}")

    lines.append(
        f"\n(A real LLM call would read the {len(chunks)} code chunk(s) above "
        f"and produce a natural-language explanation of how they answer the question.)"
    )
    return "\n".join(lines)
