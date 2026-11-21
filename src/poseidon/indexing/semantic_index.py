"""Cached, incrementally updated local semantic code index using FastEmbed (ONNX)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field

from poseidon.indexing.chunker import CodeChunk, chunk_source_file
from poseidon.patching.unified_diff import sha256_text
from poseidon.workspace import WorkspaceGuard

CACHE_DIR_NAME = ".poseidon_cache"
INDEX_FILENAME = "semantic_index.json"


class EmbeddingProvider(Protocol):
    """Protocol for any local text/code embedding backend."""

    @property
    def model_name(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedProvider:
    """Lazy-loaded local ONNX embedding provider powered by fastembed."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            kwargs: dict[str, Any] = {"model_name": self._model_name}
            if self._cache_dir:
                kwargs["cache_dir"] = self._cache_dir
            self._model = TextEmbedding(**kwargs)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = list(model.embed(texts))
        return [vec.tolist() for vec in vectors]


class IndexedChunk(BaseModel):
    """Serialized chunk + unit-normalized embedding stored in the workspace cache."""

    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    embedding: list[float]

    @property
    def qualified_id(self) -> str:
        return f"{self.path}::{self.symbol}"


class IndexSyncStats(BaseModel):
    """Summary of an incremental index sync operation."""

    model_name: str
    added_files: list[str] = Field(default_factory=list)
    updated_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    total_files: int = 0
    total_chunks: int = 0
    cache_hit: bool = False
    elapsed_ms: float = 0.0


class SemanticHit(BaseModel):
    """Single semantic code search match with path, line range, symbol, and score."""

    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    score: float

    @property
    def qualified_id(self) -> str:
        return f"{self.path}::{self.symbol}"


class SemanticSearchResult(BaseModel):
    """Structured result of a semantic code search."""

    query: str
    model_name: str
    hits: list[SemanticHit] = Field(default_factory=list)
    lookup_ms: float = 0.0
    sync_stats: IndexSyncStats

    def format_for_model(self) -> str:
        """Format semantic matches with exact path + line ranges for read_lines."""
        if not self.hits:
            return f"No semantic code matches found for: {self.query!r}"
        lines = [
            f"Semantic search ({self.model_name}, lookup {self.lookup_ms:.1f}ms) "
            f"for {self.query!r}:"
        ]
        for idx, hit in enumerate(self.hits, start=1):
            lines.append(
                f"  {idx}. {hit.path}:{hit.start_line}-{hit.end_line} "
                f"({hit.kind} {hit.symbol}, score={hit.score:.3f})"
            )
        lines.append("Tip: call read_lines(path, start_line, end_line) to inspect matching code.")
        return "\n".join(lines)


def _normalize(vector: list[float]) -> list[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return arr.tolist()


class SemanticCodeIndex:
    """Workspace-scoped, incrementally cached semantic code search index."""

    def __init__(
        self,
        workspace: WorkspaceGuard,
        provider: EmbeddingProvider,
        *,
        cache_path: Path | None = None,
    ) -> None:
        self._workspace = workspace
        self._provider = provider
        self._cache_path = cache_path or (workspace.root / CACHE_DIR_NAME / INDEX_FILENAME)
        self._file_hashes: dict[str, str] = {}
        self._chunks: list[IndexedChunk] = []
        self._loaded = False

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    def _load_cache(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._cache_path.is_file():
            return
        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("model_name") != self._provider.model_name:
            # Model changed -> rebuild cleanly
            return
        self._file_hashes = dict(payload.get("file_hashes") or {})
        self._chunks = [IndexedChunk(**item) for item in (payload.get("chunks") or [])]

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self._provider.model_name,
            "file_hashes": self._file_hashes,
            "chunks": [c.model_dump() for c in self._chunks],
        }
        self._cache_path.write_text(json.dumps(payload), encoding="utf-8")

    def sync(self) -> IndexSyncStats:
        """Incrementally update the cached index for added, edited, or deleted workspace files."""
        started = time.perf_counter()
        self._load_cache()

        current_files = set(self._workspace.list_files())
        cached_files = set(self._file_hashes.keys())

        deleted_files = sorted(cached_files - current_files)
        added_files: list[str] = []
        updated_files: list[str] = []
        files_to_embed: list[tuple[str, str, str]] = []

        for rel_path in sorted(current_files):
            try:
                content = self._workspace.read_text(rel_path)
            except (OSError, UnicodeDecodeError):
                continue
            digest = sha256_text(content)
            if rel_path not in self._file_hashes:
                added_files.append(rel_path)
                files_to_embed.append((rel_path, content, digest))
            elif self._file_hashes[rel_path] != digest:
                updated_files.append(rel_path)
                files_to_embed.append((rel_path, content, digest))

        dirty_paths = set(deleted_files) | set(updated_files)
        if dirty_paths:
            self._chunks = [c for c in self._chunks if c.path not in dirty_paths]
            for path in deleted_files:
                self._file_hashes.pop(path, None)

        if files_to_embed:
            new_raw_chunks: list[CodeChunk] = []
            for rel_path, content, digest in files_to_embed:
                new_raw_chunks.extend(chunk_source_file(rel_path, content))
                self._file_hashes[rel_path] = digest

            if new_raw_chunks:
                texts = [chunk.to_embedding_text() for chunk in new_raw_chunks]
                raw_vectors = self._provider.embed(texts)
                for chunk, vec in zip(new_raw_chunks, raw_vectors, strict=True):
                    self._chunks.append(
                        IndexedChunk(
                            path=chunk.path,
                            symbol=chunk.symbol,
                            kind=chunk.kind,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            embedding=_normalize(vec),
                        )
                    )

        changed = bool(added_files or updated_files or deleted_files)
        if changed or not self._cache_path.is_file():
            self._save_cache()

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return IndexSyncStats(
            model_name=self._provider.model_name,
            added_files=added_files,
            updated_files=updated_files,
            deleted_files=deleted_files,
            total_files=len(self._file_hashes),
            total_chunks=len(self._chunks),
            cache_hit=not changed,
            elapsed_ms=elapsed_ms,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        auto_sync: bool = True,
        exclude_tests: bool = False,
    ) -> SemanticSearchResult:
        """Search indexed functions/blocks by semantic meaning and return path + line ranges."""
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Semantic search query cannot be empty.")

        sync_stats = self.sync() if auto_sync else IndexSyncStats(
            model_name=self._provider.model_name,
            total_files=len(self._file_hashes),
            total_chunks=len(self._chunks),
            cache_hit=True,
            elapsed_ms=0.0,
        )

        # Measure lookup time strictly AFTER indexing
        lookup_started = time.perf_counter()
        def _is_test(path_str: str) -> bool:
            p = Path(path_str)
            return p.name.startswith("test_") or "tests" in p.parts

        candidates = [
            c for c in self._chunks if not (exclude_tests and _is_test(c.path))
        ]
        if not candidates:
            return SemanticSearchResult(
                query=cleaned,
                model_name=self._provider.model_name,
                hits=[],
                lookup_ms=(time.perf_counter() - lookup_started) * 1000.0,
                sync_stats=sync_stats,
            )

        query_vec = np.asarray(_normalize(self._provider.embed([cleaned])[0]), dtype=np.float32)
        matrix = np.asarray([c.embedding for c in candidates], dtype=np.float32)
        scores = matrix @ query_vec

        ranked_indices = np.argsort(scores)[::-1][: max(1, top_k)]
        hits = [
            SemanticHit(
                path=candidates[int(idx)].path,
                symbol=candidates[int(idx)].symbol,
                kind=candidates[int(idx)].kind,
                start_line=candidates[int(idx)].start_line,
                end_line=candidates[int(idx)].end_line,
                score=float(scores[int(idx)]),
            )
            for idx in ranked_indices
        ]
        lookup_ms = (time.perf_counter() - lookup_started) * 1000.0

        return SemanticSearchResult(
            query=cleaned,
            model_name=self._provider.model_name,
            hits=hits,
            lookup_ms=lookup_ms,
            sync_stats=sync_stats,
        )
