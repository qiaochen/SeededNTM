"""ProvenanceLogger — structured audit trail for marker agent decisions.

Records database queries, LLM calls, scoring decisions, timing, and errors
in a structured JSON format suitable for publication supplementary materials.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueryRecord:
    """Record of a single database query."""

    source: str
    cell_type: str
    query_params: Dict[str, Any] = field(default_factory=dict)
    n_results: int = 0
    top_hits: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class LLMCallRecord:
    """Record of a single LLM invocation."""

    purpose: str
    input_summary: str
    output_summary: str
    elapsed_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ErrorRecord:
    """Record of a source failure or fallback."""

    source: str
    cell_type: str
    error_type: str
    message: str
    fallback_used: Optional[str] = None


class ProvenanceLogger:
    """Accumulates provenance records during a marker agent run.

    Usage:
        prov = ProvenanceLogger(context_dict)
        with prov.timed_query("PanglaoDB", "CAF", params) as rec:
            hits = source.search(...)
            rec.n_results = len(hits)
            rec.top_hits = [h.gene_symbol for h in hits[:10]]
        ...
        prov.save(output_path)
    """

    def __init__(self, context: Dict[str, Any]):
        self._start_time = time.time()
        self._context = context
        self._queries: List[QueryRecord] = []
        self._llm_calls: List[LLMCallRecord] = []
        self._errors: List[ErrorRecord] = []
        self._scoring: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._sources_queried: set = set()

    def record_query(
        self,
        source: str,
        cell_type: str,
        query_params: Dict[str, Any],
        n_results: int,
        top_hits: List[str],
        elapsed_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a completed database query."""
        self._sources_queried.add(source)
        self._queries.append(
            QueryRecord(
                source=source,
                cell_type=cell_type,
                query_params=query_params,
                n_results=n_results,
                top_hits=top_hits[:10],
                elapsed_ms=round(elapsed_ms, 1),
                error=error,
            )
        )

    def record_llm_call(
        self,
        purpose: str,
        input_summary: str,
        output_summary: str,
        elapsed_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a completed LLM invocation."""
        self._llm_calls.append(
            LLMCallRecord(
                purpose=purpose,
                input_summary=_truncate(input_summary, 200),
                output_summary=_truncate(output_summary, 300),
                elapsed_ms=round(elapsed_ms, 1),
                error=error,
            )
        )

    def record_error(
        self,
        source: str,
        cell_type: str,
        error_type: str,
        message: str,
        fallback_used: Optional[str] = None,
    ) -> None:
        """Record a source failure or fallback event."""
        self._errors.append(
            ErrorRecord(
                source=source,
                cell_type=cell_type,
                error_type=error_type,
                message=_truncate(str(message), 300),
                fallback_used=fallback_used,
            )
        )

    def record_scoring(
        self,
        cell_type: str,
        gene_scores: Dict[str, Dict[str, float]],
    ) -> None:
        """Record per-gene scoring breakdown for a cell type.

        Args:
            cell_type: The cell type being scored.
            gene_scores: Dict of gene -> {source: weighted_score, ..., "total": total}.
        """
        self._scoring[cell_type] = gene_scores

    class _TimedQuery:
        """Context manager for timing a database query."""

        def __init__(self, logger: "ProvenanceLogger", source: str, cell_type: str, params: Dict[str, Any]):
            self._logger = logger
            self._source = source
            self._cell_type = cell_type
            self._params = params
            self.n_results = 0
            self.top_hits: List[str] = []
            self._error: Optional[str] = None
            self._start: float = 0.0

        def __enter__(self):
            self._start = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed_ms = (time.time() - self._start) * 1000
            if exc_type is not None:
                self._error = f"{exc_type.__name__}: {exc_val}"
            self._logger.record_query(
                source=self._source,
                cell_type=self._cell_type,
                query_params=self._params,
                n_results=self.n_results,
                top_hits=self.top_hits,
                elapsed_ms=elapsed_ms,
                error=self._error,
            )
            return False

    def timed_query(self, source: str, cell_type: str, params: Optional[Dict[str, Any]] = None):
        """Context manager that records a timed database query.

        Example:
            with prov.timed_query("PanglaoDB", "CAF", {"organ": "Colon"}) as rec:
                hits = source.search(...)
                rec.n_results = len(hits)
                rec.top_hits = [h.gene_symbol for h in hits[:10]]
        """
        return self._TimedQuery(self, source, cell_type, params or {})

    class _TimedLLMCall:
        """Context manager for timing an LLM call."""

        def __init__(self, logger: "ProvenanceLogger", purpose: str, input_summary: str):
            self._logger = logger
            self._purpose = purpose
            self._input_summary = input_summary
            self.output_summary = ""
            self._error: Optional[str] = None
            self._start: float = 0.0

        def __enter__(self):
            self._start = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed_ms = (time.time() - self._start) * 1000
            if exc_type is not None:
                self._error = f"{exc_type.__name__}: {exc_val}"
            self._logger.record_llm_call(
                purpose=self._purpose,
                input_summary=self._input_summary,
                output_summary=self.output_summary,
                elapsed_ms=elapsed_ms,
                error=self._error,
            )
            return False

    def timed_llm_call(self, purpose: str, input_summary: str):
        """Context manager that records a timed LLM invocation.

        Example:
            with prov.timed_llm_call("nomenclature_resolution", "Resolve 'CAF'") as rec:
                result = llm_complete(...)
                rec.output_summary = result[:100]
        """
        return self._TimedLLMCall(self, purpose, input_summary)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full provenance log to a dictionary."""
        total_runtime = time.time() - self._start_time
        return {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "context": self._context,
                "sources_queried": sorted(self._sources_queried),
                "total_runtime_seconds": round(total_runtime, 2),
            },
            "queries": [asdict(q) for q in self._queries],
            "llm_calls": [asdict(c) for c in self._llm_calls],
            "scoring": self._scoring,
            "errors": [asdict(e) for e in self._errors],
        }

    def save(self, output_path: str) -> str:
        """Save provenance log as JSON.

        Args:
            output_path: Path to write the JSON file.

        Returns:
            The absolute path of the written file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return str(path.resolve())


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, appending '...' if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
