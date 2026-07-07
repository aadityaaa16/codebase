"""
run_indexer.py

Walks a directory tree, finds every .py file, and runs the chunker
on each one. This is the "index a whole repo" step, before we add
embeddings on top.
"""

import os
from chunker import chunk_file, CodeChunk
from typing import List


def find_python_files(root_dir: str) -> List[str]:
    """
    Recursively finds all .py files under root_dir, skipping common
    noise directories (virtual envs, caches, git internals).
    """
    skip_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules"}
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if fname.endswith(".py"):
                py_files.append(os.path.join(dirpath, fname))
    return sorted(py_files)


def index_repo(root_dir: str) -> List[CodeChunk]:
    """
    Chunks every Python file in root_dir and returns the full list
    of CodeChunks across the entire repo.
    """
    all_chunks: List[CodeChunk] = []
    files = find_python_files(root_dir)

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            chunks = chunk_file(file_path, source)
            all_chunks.extend(chunks)
        except SyntaxError as e:
            # Real repos sometimes have files that don't parse cleanly
            # (e.g. Python 2 code, generated files). We skip and report
            # rather than crashing the whole indexing run.
            print(f"  [SKIPPED - syntax error] {file_path}: {e}")

    return all_chunks


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "sample_repo"

    chunks = index_repo(root)

    print(f"\nIndexed {len(chunks)} chunks from repo: {root}\n")
    print(f"{'TYPE':<10} {'NAME':<28} {'PARENT':<15} {'FILE'}")
    print("-" * 90)
    for c in chunks:
        parent = c.parent_class or "-"
        print(f"{c.chunk_type:<10} {c.name:<28} {parent:<15} {c.file_path}")
