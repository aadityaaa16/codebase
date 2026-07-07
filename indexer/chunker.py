"""
chunker.py

Splits a Python file into chunks aligned with its actual structure
(functions, classes, methods) instead of arbitrary line/character splits.

Why this matters:
A naive text splitter might cut a function definition in half, or merge
two unrelated functions into one chunk. That destroys the meaning GPT
needs to answer questions like "where is JWT created?" accurately.

By using Python's `ast` module, we get exact start/end line numbers for
every function and class, so each chunk is a complete, meaningful unit
of code.
"""

import ast
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CodeChunk:
    file_path: str
    chunk_type: str          # "function", "class", or "method"
    name: str                # e.g. "create_access_token"
    parent_class: Optional[str]  # e.g. "EmailService" if this is a method
    start_line: int
    end_line: int
    code: str                # the actual source code of this chunk
    docstring: Optional[str] = None
    imports: List[str] = field(default_factory=list)


def _get_imports(tree: ast.Module) -> List[str]:
    """
    Collects top-level import statements from the file.
    We attach these to every chunk because a function's meaning often
    depends on what it imports (e.g. `from auth.jwt import create_access_token`
    tells us this file is related to auth even if the function name doesn't).
    """
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    return imports


def _get_source_segment(source_lines: List[str], node: ast.AST) -> str:
    """
    Extracts the exact source code text for a given AST node,
    using its line number range.
    """
    start = node.lineno - 1
    end = node.end_lineno
    return "\n".join(source_lines[start:end])


def _extract_module_level_chunk(
    file_path: str, source_lines: List[str], tree: ast.Module, imports: List[str]
) -> Optional[CodeChunk]:
    """
    Captures everything at the top level of the file that is NOT a
    function or class: constants, config values, top-level expressions,
    `if __name__ == "__main__"` blocks, etc.

    Why this matters: real answers sometimes live here, not inside any
    function. Example: `ALGORITHM = "HS256"` answers "what algorithm
    signs the JWT?" but no function chunk contains that fact. Skipping
    this would silently make such questions unanswerable.

    We collect these into ONE chunk per file (not one per line) because
    individually they're too small to be meaningful chunks on their own,
    but together they represent "the file's configuration/context".
    """
    leftover_nodes = [
        node for node in ast.iter_child_nodes(tree)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        # skip plain imports, since they're already captured separately in `imports`
        and not isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    if not leftover_nodes:
        return None

    lines_used = [_get_source_segment(source_lines, node) for node in leftover_nodes]
    combined_code = "\n".join(lines_used).strip()
    if not combined_code:
        return None

    start_line = min(n.lineno for n in leftover_nodes)
    end_line = max(n.end_lineno for n in leftover_nodes)

    return CodeChunk(
        file_path=file_path,
        chunk_type="module",
        name=file_path.split("/")[-1],  # use filename as the "name" for this chunk
        parent_class=None,
        start_line=start_line,
        end_line=end_line,
        code=combined_code,
        docstring=ast.get_docstring(tree),
        imports=imports,
    )


def chunk_file(file_path: str, source_code: str) -> List[CodeChunk]:
    """
    Parses a Python source file and returns a list of CodeChunks,
    one per top-level function, class, method, and one module-level
    chunk for constants/config that live outside any function or class.
    """
    tree = ast.parse(source_code)
    source_lines = source_code.splitlines()
    imports = _get_imports(tree)
    chunks: List[CodeChunk] = []

    module_chunk = _extract_module_level_chunk(file_path, source_lines, tree, imports)
    if module_chunk:
        chunks.append(module_chunk)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(CodeChunk(
                file_path=file_path,
                chunk_type="function",
                name=node.name,
                parent_class=None,
                start_line=node.lineno,
                end_line=node.end_lineno,
                code=_get_source_segment(source_lines, node),
                docstring=ast.get_docstring(node),
                imports=imports,
            ))

        elif isinstance(node, ast.ClassDef):
            # First, capture the class itself as one chunk (useful for
            # "what does this class do overall" questions)
            chunks.append(CodeChunk(
                file_path=file_path,
                chunk_type="class",
                name=node.name,
                parent_class=None,
                start_line=node.lineno,
                end_line=node.end_lineno,
                code=_get_source_segment(source_lines, node),
                docstring=ast.get_docstring(node),
                imports=imports,
            ))

            # Then, also chunk each method individually. This matters because
            # a class can have many methods, and a question like "how does
            # password verification work?" should retrieve just that method,
            # not the entire class.
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        chunk_type="method",
                        name=child.name,
                        parent_class=node.name,
                        start_line=child.lineno,
                        end_line=child.end_lineno,
                        code=_get_source_segment(source_lines, child),
                        docstring=ast.get_docstring(child),
                        imports=imports,
                    ))

    return chunks


if __name__ == "__main__":
    # Quick manual test against one file, so we can eyeball the output
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_repo/auth/jwt.py"
    with open(path) as f:
        code = f.read()

    result = chunk_file(path, code)
    for chunk in result:
        print("=" * 60)
        print(f"[{chunk.chunk_type}] {chunk.name}  (lines {chunk.start_line}-{chunk.end_line})")
        if chunk.parent_class:
            print(f"  parent class: {chunk.parent_class}")
        print(f"  docstring: {chunk.docstring}")
        print("-" * 60)
        print(chunk.code)
