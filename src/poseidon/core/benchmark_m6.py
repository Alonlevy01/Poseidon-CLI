"""Milestone 6 benchmark: 4-question comparison of grep (ripgrep) vs. local semantic embeddings."""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from poseidon.indexing.chunker import resolve_enclosing_symbol
from poseidon.indexing.semantic_index import SemanticCodeIndex
from poseidon.search.lexical import search_workspace_text
from poseidon.workspace import WorkspaceGuard


@dataclass(frozen=True)
class LookupQuestion:
    """One Milestone 6 lookup benchmark question with pre-marked ground-truth answer."""

    id: str
    kind: str  # "exact" or "plain-english"
    question: str
    expected_target: str  # "path::symbol"
    grep_query: str
    semantic_query: str


class RankedMatch(BaseModel):
    """Distinct function or code block returned in Top-3 by either search method."""

    qualified_id: str
    location: str
    detail: str


class QuestionBenchmarkResult(BaseModel):
    """Side-by-side grep vs. semantic search outcome for one question."""

    id: str
    kind: str
    question: str
    expected_target: str
    grep_query: str
    semantic_query: str
    grep_top3: list[RankedMatch] = Field(default_factory=list)
    semantic_top3: list[RankedMatch] = Field(default_factory=list)
    grep_hit: bool
    semantic_hit: bool
    grep_ms: float
    semantic_ms: float


class SearchBenchmarkReport(BaseModel):
    """Aggregate Milestone 6 benchmark summary."""

    model_name: str
    indexing_ms: float
    indexed_chunks: int
    results: list[QuestionBenchmarkResult]
    grep_hits: int
    semantic_hits: int
    total_questions: int
    grep_total_ms: float
    semantic_total_ms: float


DEFAULT_DEMO_QUESTIONS: list[LookupQuestion] = [
    LookupQuestion(
        id="Q1",
        kind="exact",
        question="Where is calculate_budget_status defined?",
        expected_target="analytics.py::calculate_budget_status",
        grep_query="calculate_budget_status",
        semantic_query="calculate_budget_status function",
    ),
    LookupQuestion(
        id="Q2",
        kind="exact",
        question="Where is format_over_budget_line defined?",
        expected_target="formatter.py::format_over_budget_line",
        grep_query="format_over_budget_line",
        semantic_query="format_over_budget_line function",
    ),
    LookupQuestion(
        id="Q3",
        kind="plain-english",
        question="Which function rejects negative or zero spending inputs when reading user data?",
        expected_target="parser.py::parse_expenses",
        grep_query="negative spending",
        semantic_query="reject negative or zero spending amounts when validating expense input",
    ),
    LookupQuestion(
        id="Q4",
        kind="plain-english",
        question="Which function drops unwanted categories from the totals before summing?",
        expected_target="analytics.py::filter_excluded_categories",
        grep_query="unwanted categories",
        semantic_query="remove unwanted excluded categories from expense totals dictionary",
    ),
]


def _grep_top3_distinct_blocks(
    workspace: WorkspaceGuard,
    query: str,
) -> tuple[list[RankedMatch], float]:
    """Run ripgrep and map hits to the first 3 distinct enclosing functions/blocks."""
    started = time.perf_counter()
    res = search_workspace_text(workspace=workspace, query=query, max_results=30)

    seen: set[str] = set()
    ranked: list[RankedMatch] = []
    file_cache: dict[str, str] = {}

    for hit in res.hits:
        if hit.path.startswith("test_") or "/test_" in hit.path:
            continue
        if hit.path not in file_cache:
            try:
                file_cache[hit.path] = workspace.read_text(hit.path)
            except OSError:
                continue
        block = resolve_enclosing_symbol(hit.path, file_cache[hit.path], hit.line_number)
        if block.qualified_id in seen:
            continue
        seen.add(block.qualified_id)
        ranked.append(
            RankedMatch(
                qualified_id=block.qualified_id,
                location=f"{block.path}:{block.start_line}-{block.end_line}",
                detail=f"line {hit.line_number}: {hit.line_text.strip()[:70]}",
            )
        )
        if len(ranked) >= 3:
            break

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return ranked, elapsed_ms


def run_search_benchmark(
    *,
    workspace: WorkspaceGuard,
    index: SemanticCodeIndex,
    questions: list[LookupQuestion] = DEFAULT_DEMO_QUESTIONS,
) -> SearchBenchmarkReport:
    """Index the workspace, warm embeddings, and compare grep vs. semantic search on 4 questions."""
    sync_stats = index.sync()

    # Warm query encoder once before measuring post-indexing lookup times
    _ = index.search("warmup", top_k=1, auto_sync=False, exclude_tests=True)

    results: list[QuestionBenchmarkResult] = []
    for q in questions:
        grep_top3, grep_ms = _grep_top3_distinct_blocks(workspace, q.grep_query)
        sem_res = index.search(
            q.semantic_query,
            top_k=3,
            auto_sync=False,
            exclude_tests=True,
        )
        sem_top3 = [
            RankedMatch(
                qualified_id=h.qualified_id,
                location=f"{h.path}:{h.start_line}-{h.end_line}",
                detail=f"score={h.score:.3f}",
            )
            for h in sem_res.hits[:3]
        ]

        grep_hit = any(m.qualified_id == q.expected_target for m in grep_top3)
        sem_hit = any(m.qualified_id == q.expected_target for m in sem_top3)

        results.append(
            QuestionBenchmarkResult(
                id=q.id,
                kind=q.kind,
                question=q.question,
                expected_target=q.expected_target,
                grep_query=q.grep_query,
                semantic_query=q.semantic_query,
                grep_top3=grep_top3,
                semantic_top3=sem_top3,
                grep_hit=grep_hit,
                semantic_hit=sem_hit,
                grep_ms=grep_ms,
                semantic_ms=sem_res.lookup_ms,
            )
        )

    return SearchBenchmarkReport(
        model_name=sync_stats.model_name,
        indexing_ms=sync_stats.elapsed_ms,
        indexed_chunks=sync_stats.total_chunks,
        results=results,
        grep_hits=sum(1 for r in results if r.grep_hit),
        semantic_hits=sum(1 for r in results if r.semantic_hit),
        total_questions=len(results),
        grep_total_ms=sum(r.grep_ms for r in results),
        semantic_total_ms=sum(r.semantic_ms for r in results),
    )
