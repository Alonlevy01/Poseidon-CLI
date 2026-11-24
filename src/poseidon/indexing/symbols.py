"""Multi-language Tree-sitter code symbol outline (classes, functions, methods, structs)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

EXT_TO_TS_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
}

_SYMBOL_NODE_TYPES: dict[str, str] = {
    "function_definition": "function",
    "async_function_definition": "function",
    "function_declaration": "function",
    "function_item": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "class_definition": "class",
    "class_declaration": "class",
    "struct_item": "struct",
    "impl_item": "impl",
    "interface_declaration": "interface",
    "type_declaration": "type",
}


@dataclass(frozen=True)
class CodeSymbol:
    """One named structural symbol in a source file with 1-based line range."""

    kind: str
    name: str
    start_line: int
    end_line: int


def _extract_python_ast(source: str) -> list[CodeSymbol]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[CodeSymbol] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            end_ln = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
            out.append(
                CodeSymbol(kind=kind, name=node.name, start_line=node.lineno, end_line=end_ln)
            )
    out.sort(key=lambda s: (s.start_line, s.end_line))
    return out


def extract_file_symbols(rel_path: str | Path, source: str) -> list[CodeSymbol]:
    """Extract structural symbols (kind, name, line range) via Tree-sitter with AST fallback."""
    suffix = Path(rel_path).suffix.lower()
    lang_name = EXT_TO_TS_LANG.get(suffix)
    if not lang_name:
        return []

    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(lang_name)  # type: ignore[arg-type]
        raw_bytes = source.encode("utf-8", errors="replace")
        tree = parser.parse(raw_bytes)
    except Exception:
        return _extract_python_ast(source) if suffix == ".py" else []

    symbols: list[CodeSymbol] = []

    def _walk(node: object) -> None:
        ntype = getattr(node, "type", "")
        kind = _SYMBOL_NODE_TYPES.get(ntype)
        if kind is not None:
            name_node = node.child_by_field_name("name")  # type: ignore[attr-defined]
            if name_node is not None:
                sym_name = raw_bytes[name_node.start_byte : name_node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                start_ln = int(node.start_point[0]) + 1  # type: ignore[attr-defined]
                end_ln = int(node.end_point[0]) + 1  # type: ignore[attr-defined]
                symbols.append(
                    CodeSymbol(kind=kind, name=sym_name, start_line=start_ln, end_line=end_ln)
                )
        for child in getattr(node, "children", []):
            _walk(child)

    _walk(tree.root_node)
    symbols.sort(key=lambda s: (s.start_line, s.end_line))
    return symbols


def format_symbols_outline(rel_path: str, symbols: list[CodeSymbol]) -> str:
    """Format extracted symbols as a compact, token-efficient outline for the LLM."""
    if not symbols:
        return f"No structural symbols found in {rel_path}."
    lines = [f"Symbols in {rel_path} ({len(symbols)} found):"]
    for sym in symbols:
        lines.append(f"  {sym.kind:9s} {sym.name} (L{sym.start_line}-L{sym.end_line})")
    return "\n".join(lines)
