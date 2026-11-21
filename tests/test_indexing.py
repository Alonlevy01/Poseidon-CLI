"""Unit tests for AST code chunking, incremental semantic index cache, and M6 benchmark."""

import json
import re
from pathlib import Path

from poseidon.core.benchmark_m6 import LookupQuestion, run_search_benchmark
from poseidon.indexing.chunker import chunk_source_file, resolve_enclosing_symbol
from poseidon.indexing.semantic_index import SemanticCodeIndex
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard


class FakeEmbeddingProvider:
    """Fast deterministic bag-of-keywords embedder for unit tests (zero model downloads)."""

    VOCAB = [
        "budget",
        "over",
        "exceeded",
        "negative",
        "positive",
        "validate",
        "parse",
        "expense",
        "exclude",
        "filter",
        "unwanted",
        "drop",
        "format",
        "currency",
        "sort",
        "amount",
    ]

    def __init__(self) -> None:
        self.embed_calls = 0

    @property
    def model_name(self) -> str:
        return "fake/test-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            tokens = set(re.findall(r"[a-z]+", text.lower()))
            vec = [1.0 if kw in tokens else 0.0 for kw in self.VOCAB]
            if not any(vec):
                vec[0] = 0.01
            vectors.append(vec)
        return vectors


def test_ast_chunker_extracts_functions_and_line_ranges() -> None:
    source = (
        '"""Demo module."""\n'
        "\n"
        "def alpha(x: int) -> int:\n"
        "    return x + 1\n"
        "\n"
        "\n"
        "class Box:\n"
        "    def area(self) -> int:\n"
        "        return 10\n"
    )
    chunks = chunk_source_file("demo.py", source)
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [
        ("alpha", 3, 4),
        ("Box.area", 8, 9),
    ]
    resolved = resolve_enclosing_symbol("demo.py", source, 4)
    assert resolved.qualified_id == "demo.py::alpha"


def test_semantic_index_caches_and_updates_on_add_edit_delete(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text(
        "parser.py",
        "def parse_expenses(raw):\n    '''Validate positive expense amounts.'''\n    return {}\n",
    )
    guard.write_text(
        "analytics.py",
        "def filter_excluded(totals):\n    '''Filter unwanted excluded items.'''\n    return {}\n",
    )
    provider = FakeEmbeddingProvider()
    index = SemanticCodeIndex(guard, provider)

    # 1. Initial sync indexes both files
    s1 = index.sync()
    assert s1.added_files == ["analytics.py", "parser.py"]
    assert not s1.cache_hit
    assert provider.embed_calls == 1
    assert index.cache_path.is_file()

    # 2. Second sync is a full cache hit (zero re-embedding)
    s2 = index.sync()
    assert s2.cache_hit
    assert provider.embed_calls == 1

    # 3. Edit analytics.py (shift function down by 3 lines) -> incremental re-index
    guard.write_text(
        "analytics.py",
        "# line 1\n# line 2\n# line 3\n"
        "def filter_excluded(totals):\n    '''Filter unwanted excluded items.'''\n    return {}\n",
    )
    res = index.search("filter unwanted excluded items", top_k=2)
    assert res.sync_stats.updated_files == ["analytics.py"]
    assert res.hits[0].qualified_id == "analytics.py::filter_excluded"
    assert res.hits[0].start_line == 4
    assert res.hits[0].end_line == 6

    # 4. Delete parser.py -> chunks removed cleanly
    (tmp_path / "parser.py").unlink()
    s4 = index.sync()
    assert s4.deleted_files == ["parser.py"]
    assert s4.total_files == 1


def test_semantic_search_tool_and_m6_benchmark(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text(
        "parser.py",
        "def parse_expenses(raw):\n    '''Validate positive expense amounts.'''\n    return {}\n",
    )
    guard.write_text(
        "analytics.py",
        (
            "def calculate_budget_status(total, budget):\n"
            "    '''Check over budget.'''\n"
            "    return {}\n"
        ),
    )
    provider = FakeEmbeddingProvider()
    index = SemanticCodeIndex(guard, provider)

    registry = ToolRegistry()
    register_core_tools(
        registry=registry,
        workspace=guard,
        state=ToolSessionState(),
        semantic_index=index,
    )
    assert "semantic_search" in registry.names()
    tool_res = registry.execute("semantic_search", json.dumps({"query": "over budget", "top_k": 2}))
    assert tool_res.ok
    assert "analytics.py:1-3" in tool_res.output

    questions = [
        LookupQuestion(
            id="Q1",
            kind="exact",
            question="Where is calculate_budget_status?",
            expected_target="analytics.py::calculate_budget_status",
            grep_query="calculate_budget_status",
            semantic_query="over budget",
        ),
        LookupQuestion(
            id="Q2",
            kind="plain-english",
            question="Validate spending?",
            expected_target="parser.py::parse_expenses",
            grep_query="no_match_terms_here",
            semantic_query="validate positive expense",
        ),
    ]
    report = run_search_benchmark(workspace=guard, index=index, questions=questions)
    assert report.total_questions == 2
    assert report.grep_hits == 1
    assert report.semantic_hits == 2
