# AI Codebase Navigator

A RAG (Retrieval-Augmented Generation) system for navigating and understanding
unfamiliar codebases through natural-language questions — instead of manually
searching through files, ask "Where is JWT authentication implemented?" and
get a grounded answer with exact file paths and line numbers.

## Why this exists

Joining a new codebase with 100k+ lines of code means spending hours grepping,
opening files, and asking senior developers questions that are often answered
by code that already exists — you just can't find it. This tool answers those
questions directly, with every answer traceable back to real source code.

## Architecture

```
Repo files (.py)
      │
      ▼
AST-based chunker  ──────► splits code by function/class/method boundaries,
      │                     not arbitrary text splits (preserves meaning)
      ▼
Embeddings (sentence-transformers, local, free)
      │
      ▼
ChromaDB (vector store)
      │
      ▼
Hybrid search ◄──────────── combines semantic similarity + BM25 keyword
      │                      matching, since neither alone is reliable
      ▼
Gemini (explanation layer) ─► turns retrieved chunks into a plain-English
      │                        answer, grounded only in what was retrieved
      ▼
FastAPI backend ────────────► /index and /query endpoints
      │
      ▼
Streamlit UI
```

## Key design decisions

- **Chunking by AST node, not text splitting.** A naive text splitter can cut
  a function definition in half. Using Python's `ast` module gives exact
  function/class boundaries, so every chunk is a complete, meaningful unit.
- **Dual-level chunking for classes.** Both the whole class *and* each
  individual method are stored separately, so a question about one method
  doesn't retrieve the entire class, but a question about the class's overall
  purpose still has that context available.
- **A dedicated "module-level" chunk per file** captures constants and config
  that live outside any function/class (e.g. `ALGORITHM = "HS256"`) — these
  answer real questions but would otherwise be invisible to retrieval.
- **Hybrid search (semantic + BM25), not either alone.** Semantic search can
  get distracted by loosely related concepts; BM25 can be fooled by chunks
  that merely *mention* a term rather than define it. Combining both with a
  tunable weight (`alpha`) is more robust than either in isolation.
- **Every answer is grounded with file path + line numbers.** The explanation
  layer is explicitly instructed to only use the retrieved chunks and cite
  their source, rather than generating plausible-sounding but ungrounded text.

## Known limitations (intentional v1 scope decisions)

- Python only (no multi-language support yet)
- No call graph / cross-file dependency graph (a real static-analysis project
  in itself — scoped out to keep this achievable and defensible)
- No confidence score (a fabricated-looking confidence percentage is worse
  than no confidence score at all, unless properly calibrated)
- Function-level chunking means a method calling a sibling method in the same
  file won't have that relationship explicitly encoded

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose your mode

**Free, fully local mock (default):** no setup needed. Uses a lightweight
TF-IDF stand-in for embeddings and template-based explanations - good for
verifying the pipeline works, not for real use.

**Free, real quality with Gemini:** get a free Gemini API key at
https://aistudio.google.com/apikey. This single key powers both the
embeddings and the explanations - no PyTorch or local model download needed:
```bash
export EMBEDDING_MODE=real
export EXPLAIN_MODE=real
export GEMINI_API_KEY=your_key_here
```

**Free, real quality with Groq (alternative explanation provider):** if
Gemini's free tier is rate-limited or giving quota errors, swap the
explanation layer to Groq - no credit card required, generous free tier,
fast responses (Gemini still handles embeddings):
```bash
export EMBEDDING_MODE=real
export EXPLAIN_MODE=groq
export GEMINI_API_KEY=your_key_here   # still needed for embeddings
export GROQ_API_KEY=your_key_here     # get one free at https://console.groq.com
```

### 3. Run the backend
```bash
cd api
uvicorn main:app --reload
```
Visit `http://localhost:8000/docs` for interactive API docs.

### 4. Run the UI (in a separate terminal)
```bash
cd ui
streamlit run app.py
```
Visit `http://localhost:8501`.

### 5. Try it
1. In the sidebar, index the included sample repo (or point it at your own
   Python project): `sample_repo/`
2. Ask questions like:
   - "Where is JWT authentication implemented?"
   - "How is password reset implemented?"
   - "Which endpoint creates a user?"

## Project structure

```
indexer/       # chunking + embedding generation
retrieval/     # BM25, hybrid search, LLM explanation
api/           # FastAPI backend
ui/            # Streamlit frontend
sample_repo/   # small test repo mirroring common auth/login patterns
```

## Tech stack

FastAPI · Streamlit · ChromaDB · Google Gemini (embeddings + explanations) ·
BM25 (`rank_bm25`) · Python `ast` module
