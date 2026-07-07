"""
app.py

Streamlit UI for the AI Codebase Navigator.

Design concept: instead of a generic chatbot skin, the source results
are styled like a "Find in Files" panel from a code editor - file
breadcrumbs, gutter-style line numbers, monospace snippets. This is
the one visual idea worth spending polish on, since it's the moment
that actually differentiates this from a bolted-on chat widget.

Run with:
    streamlit run app.py
(requires the FastAPI backend running at localhost:8000 - see api/main.py)
"""

import streamlit as st
import requests

import os as _os
API_BASE = "http://localhost:8000"
_DEFAULT_SAMPLE_REPO = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "sample_repo")
)

st.set_page_config(
    page_title="Codebase Navigator",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM DESIGN SYSTEM
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #14171C;
    --surface: #1C2128;
    --surface-hover: #232933;
    --border: #2A303B;
    --accent: #E8A33D;
    --accent-dim: #8A6423;
    --teal: #5FB3B3;
    --text: #E6E9EF;
    --text-muted: #8B93A1;
    --text-faint: #5A6270;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: var(--bg);
}

/* Hide Streamlit's default chrome so our design owns the whole page */
#MainMenu, footer, header {visibility: hidden;}

/* --- Header --- */
.nav-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.2rem;
}
.nav-header .prompt-glyph {
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
}
.cursor-blink {
    display: inline-block;
    width: 0.5rem;
    height: 1.1rem;
    background: var(--accent);
    margin-left: 0.4rem;
    vertical-align: middle;
    animation: blink 1.1s steps(1) infinite;
}
@media (prefers-reduced-motion: reduce) {
    .cursor-blink { animation: none; opacity: 0.6; }
}
@keyframes blink {
    0%, 49% { opacity: 1; }
    50%, 100% { opacity: 0; }
}
.nav-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.6rem;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.01em;
}
.nav-subtitle {
    font-family: 'Inter', sans-serif;
    color: var(--text-muted);
    font-size: 0.92rem;
    margin-bottom: 1.6rem;
    margin-left: 2.2rem;
}

/* --- Status badges --- */
.status-row { display: flex; gap: 0.5rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    border: 1px solid var(--border);
    color: var(--text-muted);
    background: var(--surface);
}
.badge.live { color: var(--teal); border-color: var(--teal); }
.badge.mock { color: var(--text-faint); }

/* --- Question block --- */
.question-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    color: var(--text);
    background: var(--surface);
    border-left: 3px solid var(--accent);
    padding: 0.8rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 1.4rem 0 0.9rem 0;
}
.question-block .qlabel {
    color: var(--accent);
    margin-right: 0.5rem;
}

/* --- Answer text --- */
.answer-block {
    font-family: 'Inter', sans-serif;
    font-size: 0.98rem;
    line-height: 1.65;
    color: var(--text);
    padding: 0 0.2rem 0.6rem 0.2rem;
}

/* --- Sources panel, styled like an editor "Find in Files" result list --- */
.sources-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
    margin: 1.1rem 0 0.5rem 0.2rem;
}
.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 0.5rem;
    overflow: hidden;
    transition: border-color 0.15s ease;
}
.source-card:hover { border-color: var(--accent-dim); }
.source-breadcrumb {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.9rem;
    background: var(--surface-hover);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}
.source-path { color: var(--text-muted); }
.source-path .filename { color: var(--text); font-weight: 500; }
.source-lines {
    color: var(--teal);
    font-size: 0.75rem;
    background: rgba(95, 179, 179, 0.1);
    padding: 0.1rem 0.45rem;
    border-radius: 3px;
}
.source-meta {
    display: flex;
    gap: 0.5rem;
    align-items: center;
}
.chunk-type-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent);
    border: 1px solid var(--accent-dim);
    padding: 0.05rem 0.4rem;
    border-radius: 3px;
}
.relevance-bar-track {
    width: 50px;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.relevance-bar-fill {
    height: 100%;
    background: var(--accent);
}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Input styling */
.stTextInput input, .stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

.stButton button {
    font-family: 'IBM Plex Mono', monospace;
    background-color: var(--accent) !important;
    color: #14171C !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton button:hover {
    background-color: #F2B860 !important;
}

.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-faint);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
}
.empty-state .glyph {
    font-size: 2rem;
    color: var(--accent-dim);
    display: block;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ---------------------------------------------------------------------------
# SIDEBAR - indexing controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Repository")
    repo_path = st.text_input(
        "Path to index",
        value=_DEFAULT_SAMPLE_REPO,
        label_visibility="collapsed",
    )
    if st.button("Index Repository", use_container_width=True):
        with st.spinner("Chunking and embedding..."):
            try:
                resp = requests.post(f"{API_BASE}/index", json={"repo_path": repo_path}, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.indexed = True
                    st.session_state.chunk_count = data["chunks_indexed"]
                    st.success(f"Indexed {data['chunks_indexed']} chunks")
                else:
                    st.error(resp.json().get("detail", "Indexing failed"))
            except requests.exceptions.ConnectionError:
                st.error("Can't reach API. Is it running on localhost:8000?")

    st.markdown("---")
    st.markdown("### System status")
    try:
        health = requests.get(f"{API_BASE}/", timeout=5).json()
        emb_mode = health.get("embedding_mode", "unknown")
        exp_mode = health.get("explain_mode", "unknown")
        st.markdown(f"""
        <div class="status-row">
            <span class="badge {'live' if emb_mode == 'real' else 'mock'}">embeddings: {emb_mode}</span>
        </div>
        <div class="status-row">
            <span class="badge {'live' if exp_mode == 'real' else 'mock'}">explain: {exp_mode}</span>
        </div>
        """, unsafe_allow_html=True)
        if health.get("chunk_count"):
            st.caption(f"{health['chunk_count']} chunks indexed from {health.get('indexed_repo', '')}")
    except requests.exceptions.ConnectionError:
        st.warning("API not reachable")

    st.markdown("---")
    with st.expander("Search settings"):
        n_results = st.slider("Number of sources", 1, 10, 5)
        alpha = st.slider("Semantic vs. keyword weight", 0.0, 1.0, 0.5, 0.1,
                           help="0 = pure keyword (BM25), 1 = pure semantic search")


# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.markdown("""
<div class="nav-header">
    <span class="prompt-glyph">&gt;</span>
    <h1>Codebase Navigator</h1>
    <span class="cursor-blink"></span>
</div>
<div class="nav-subtitle">Ask questions about your codebase, get grounded answers with file + line references.</div>
""", unsafe_allow_html=True)

question = st.text_input(
    "Ask a question",
    placeholder="Where is JWT authentication implemented?",
    label_visibility="collapsed",
)
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Searching..."):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={"question": question, "n_results": n_results, "alpha": alpha},
                timeout=60,
            )
            if resp.status_code == 200:
                st.session_state.history.insert(0, resp.json())
            else:
                st.error(resp.json().get("detail", "Query failed"))
        except requests.exceptions.ConnectionError:
            st.error("Can't reach API. Is it running on localhost:8000?")

if not st.session_state.history:
    st.markdown("""
    <div class="empty-state">
        <span class="glyph">◆</span>
        Index a repository, then ask your first question.
    </div>
    """, unsafe_allow_html=True)

for item in st.session_state.history:
    st.markdown(f"""
    <div class="question-block"><span class="qlabel">?</span>{item['question']}</div>
    <div class="answer-block">{item['answer']}</div>
    """, unsafe_allow_html=True)

    if item["sources"]:
        st.markdown('<div class="sources-label">Source files</div>', unsafe_allow_html=True)
        for src in item["sources"]:
            full_path = src["file_path"].replace("\\", "/")  # normalize Windows backslashes for display
            filename = _os.path.basename(full_path)
            folder = _os.path.dirname(full_path)
            # Show path relative to the indexed repo root, not the full absolute path,
            # by trimming everything up through the last folder that matches the
            # indexed repo's own folder name (works for any repo, not just sample_repo)
            repo_root_name = _os.path.basename(_os.path.normpath(repo_path)) if 'repo_path' in dir() else None
            if repo_root_name and repo_root_name in folder:
                folder = folder.split(repo_root_name, 1)[-1].lstrip("/")
            relevance_pct = min(100, max(5, int(src["hybrid_score"] * 100)))
            parent = f"{src.get('chunk_type', '')}"
            st.markdown(f"""
            <div class="source-card">
                <div class="source-breadcrumb">
                    <div class="source-path">{folder}/<span class="filename">{filename}</span></div>
                    <div class="source-meta">
                        <span class="chunk-type-tag">{src['chunk_type']}</span>
                        <span class="source-lines">L{src['start_line']}-{src['end_line']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
