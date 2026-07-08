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
Embeddings (Gemini API, free tier)
      │
      ▼
ChromaDB (vector store)
      │
      ▼
Hybrid search ◄──────────── combines semantic similarity + BM25 keyword
      │                      matching, since neither alone is reliable
      ▼
Gemini or Groq (explanation layer) ─► turns retrieved chunks into a plain-English
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
- **GitHub URL indexing, not just local paths.** A locally-run instance can
  index any folder on your disk. But a publicly deployed instance has no
  access to a visitor's filesystem - so it can also clone any public GitHub
  repo server-side (shallow clone, validated to be a real github.com HTTPS
  URL only) and index that instead.

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
1. In the sidebar, either paste a public GitHub URL to index any repo
   (e.g. `https://github.com/navdeep-G/samplemod`), or switch to
   "Local path" to index the included `sample_repo/` or your own
   Python project on disk.
2. Ask questions like:
   - "Where is JWT authentication implemented?"
   - "How is password reset implemented?"
   - "Which endpoint creates a user?"

## Deploying publicly (Railway + Streamlit Community Cloud)

This deploys the FastAPI backend on Railway and the Streamlit UI on
Streamlit Community Cloud, so anyone can use it from a browser without
installing anything.

### 1. Push to GitHub first
Both platforms deploy directly from a GitHub repo - see the Git setup
section below if you haven't pushed yet.

### 2. Deploy the backend on Railway
1. Go to https://railway.app, create a new project, choose
   "Deploy from GitHub repo", and select this repo.
2. Railway will detect `railway.json` and use it automatically - no
   extra configuration needed for the start command.
3. In the project's **Variables** tab, add:
   ```
   EMBEDDING_MODE=real
   EXPLAIN_MODE=groq
   GEMINI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```
4. Once deployed, click **Settings > Networking > Generate Domain** to
   get a public URL like `https://your-app.up.railway.app`. Copy it -
   you'll need it in the next step.
5. Test it by visiting `https://your-app.up.railway.app/` in a browser -
   you should see the same JSON health check as locally.

### 3. Deploy the UI on Streamlit Community Cloud
1. Go to https://share.streamlit.io, click "Create app", and connect
   your GitHub repo.
2. Set the file path to `ui/app.py`.
3. Before deploying, open "Advanced settings" and paste this into the
   **Secrets** field (replacing the URL with your actual Railway URL
   from step 2):
   ```toml
   API_BASE = "https://your-app.up.railway.app"
   ```
4. Click Deploy. You'll get a public URL like
   `https://your-app.streamlit.app` that anyone can open directly.

### Known limitations of the public deployment
- ChromaDB's index storage is ephemeral on Railway's free tier - it
  resets on redeploy, which is fine for a demo but not for a
  production system with real user data.
- Only one repo is indexed at a time (global, not per-visitor) - if
  two people use the demo simultaneously, the second person's indexing
  request replaces the first person's index. Acceptable for a portfolio
  demo, not for real multi-user use.
- Cloning very large repos may be slow or hit Railway's free-tier
  resource limits - the shallow (depth=1) clone helps, but there's no
  hard size cap enforced.



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
Groq (alternative explanation provider) · BM25 (`rank_bm25`) ·
Python `ast` module · GitPython · Railway · Streamlit Community Cloud
