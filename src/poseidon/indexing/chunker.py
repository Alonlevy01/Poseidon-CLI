"""AST-aware Python code chunker with sliding-window fallback for non-Python files."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeChunk:
    """A single indexed function, method, class, or code block with exact line bounds."""

    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    source: str

    @property
    def qualified_id(self) -> str:
        return f"{self.path}::{self.symbol}"

    def to_embedding_text(self) -> str:
        """Build rich text input combining file path, symbol name, and source code."""
        return (
            f"File: {self.path} | {self.kind}: {self.symbol} "
            f"(lines {self.start_line}-{self.end_line})\n{self.source}"
        )


def chunk_source_file(rel_path: str, content: str) -> list[CodeChunk]:
    """Split a source file into AST functions/methods/classes, or fixed line windows."""
    lines = content.splitlines()
    if not lines:
        return []

    if rel_path.endswith(".py"):
        chunks = _chunk_python_ast(rel_path, content, lines)
        if chunks:
            return chunks

    return _chunk_line_windows(rel_path, lines)


def _chunk_python_ast(rel_path: str, content: str, lines: list[str]) -> list[CodeChunk]:
    try:
        tree = ast.parse(content, filename=rel_path)
    except SyntaxError:
        return []

    chunks: list[CodeChunk] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                _node_to_chunk(
                    rel_path=rel_path,
                    symbol=node.name,
                    kind="function",
                    node=node,
                    lines=lines,
                )
            )
        elif isinstance(node, ast.ClassDef):
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                for method in methods:
                    chunks.append(
                        _node_to_chunk(
                            rel_path=rel_path,
                            symbol=f"{node.name}.{method.name}",
                            kind="method",
                            node=method,
                            lines=lines,
                        )
                    )
            else:
                chunks.append(
                    _node_to_chunk(
                        rel_path=rel_path,
                        symbol=node.name,
                        kind="class",
                        node=node,
                        lines=lines,
                    )
                )

    if not chunks:
        # Top-level script with no functions/classes -> index as module block
        return [
            CodeChunk(
                path=rel_path,
                symbol="<module>",
                kind="module",
                start_line=1,
                end_line=len(lines),
                source="\n".join(lines),
            )
        ]

    return chunks


def _node_to_chunk(
    *,
    rel_path: str,
    symbol: str,
    kind: str,
    node: ast.stmt,
    lines: list[str],
) -> CodeChunk:
    start_line = getattr(node, "lineno", 1)
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        start_line = min(start_line, min(d.lineno for d in decorators))
    end_line = getattr(node, "end_lineno", None) or start_line
    snippet = "\n".join(lines[start_line - 1 : end_line])
    return CodeChunk(
        path=rel_path,
        symbol=symbol,
        kind=kind,
        start_line=start_line,
        end_line=end_line,
        source=snippet,
    )


def _chunk_line_windows(
    rel_path: str,
    lines: list[str],
    *,
    window_size: int = 25,
    overlap: int = 5,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    step = max(1, window_size - overlap)
    total = len(lines)
    idx = 0
    block_num = 1
    while idx < total:
        end = min(total, idx + window_size)
        snippet = "\n".join(lines[idx:end])
        chunks.append(
            CodeChunk(
                path=rel_path,
                symbol=f"block_{block_num}",
                kind="block",
                start_line=idx + 1,
                end_line=end,
                source=snippet,
            )
        )
        if end >= total:
            break
        idx += step
        block_num += 1
    return chunks


def resolve_enclosing_symbol(rel_path: str, content: str, line_number: int) -> CodeChunk:
    """Map any 1-based line number in a file to its enclosing AST function/block."""
    chunks = chunk_source_file(rel_path, content)
    for chunk in chunks:
        if chunk.start_line <= line_number <= chunk.end_line:
            return chunk
    lines = content.splitlines()
    return CodeChunk(
        path=rel_path,
        symbol="<module>",
        kind="module",
        start_line=1,
        end_line=max(1, len(lines)),
        source="\n".join(lines[max(0, line_number - 3) : min(len(lines), line_number + 3)]),
    )
