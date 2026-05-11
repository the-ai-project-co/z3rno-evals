"""EvalRunner — drives a GoldenDataset against a live z3rno-server.

Talks to the server through the public ``z3rno`` Python SDK; no
direct DB / engine access. Per-item:

  1. Call ``recall()`` with the item's strategy + top_k.
  2. Record latency.
  3. Compute recall@k, MRR, entity coverage.
  4. Invoke the faithfulness judge.

Aggregates into a :class:`RunResult` consumed by the report module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from z3rno_evals.dataset import GoldenDataset, GoldenItem
from z3rno_evals.judges.base import FaithfulnessJudge, JudgeVerdict
from z3rno_evals.judges.stub import StubJudge
from z3rno_evals.metrics import (
    LatencyPercentiles,
    entity_coverage,
    latency_percentiles,
    mrr,
    recall_at_k,
)


@dataclass(frozen=True)
class ItemResult:
    """Per-item eval outcome."""

    item_id: str
    strategy_used: str
    retrieved_ids: tuple[str, ...]
    latency_ms: float
    recall_at_k_score: float
    mrr_score: float
    entity_coverage_score: float
    faithfulness: JudgeVerdict
    within_latency_budget: bool
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    """Top-level run aggregate consumed by the report renderer."""

    dataset_name: str
    dataset_version: int
    server_url: str
    items: tuple[ItemResult, ...]
    latency: LatencyPercentiles
    # Aggregate numeric metrics — NaN means "no item contributed signal".
    mean_recall_at_k: float
    mean_mrr: float
    mean_entity_coverage: float
    mean_faithfulness: float
    items_within_budget: int
    items_total: int
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _mean_ignoring_nan(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


class EvalRunner:
    """Run a GoldenDataset against a live z3rno-server."""

    def __init__(
        self,
        *,
        client: Any,
        judge: FaithfulnessJudge | None = None,
    ) -> None:
        """``client`` is a ``z3rno.Z3rnoClient`` (sync) instance.

        We type it as ``Any`` to keep z3rno-evals importable without
        the SDK installed (it's a runtime dependency, not a build-time
        one — useful for unit tests with a fake client).
        """
        self._client = client
        self._judge = judge or StubJudge()

    def run(self, dataset: GoldenDataset, *, server_url: str) -> RunResult:
        item_results = [self._run_one(item) for item in dataset.items]

        latencies = [r.latency_ms for r in item_results if r.error is None]
        latency = latency_percentiles(latencies)

        return RunResult(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            server_url=server_url,
            items=tuple(item_results),
            latency=latency,
            mean_recall_at_k=_mean_ignoring_nan([r.recall_at_k_score for r in item_results]),
            mean_mrr=_mean_ignoring_nan([r.mrr_score for r in item_results]),
            mean_entity_coverage=_mean_ignoring_nan(
                [r.entity_coverage_score for r in item_results]
            ),
            mean_faithfulness=_mean_ignoring_nan([r.faithfulness.score for r in item_results]),
            items_within_budget=sum(1 for r in item_results if r.within_latency_budget),
            items_total=len(item_results),
        )

    # ---- internals -----

    def _run_one(self, item: GoldenItem) -> ItemResult:
        start = time.perf_counter()
        retrieved_ids: list[str] = []
        strategy_used = item.strategy
        contents: list[str] = []
        err: str | None = None

        try:
            response = self._client.recall(
                agent_id=item.agent_id,
                query=item.query,
                top_k=item.top_k,
                strategy=item.strategy,
            )
            for r in self._iter_results(response):
                rid = getattr(r, "memory_id", None) or getattr(r, "id", None)
                if rid is not None:
                    retrieved_ids.append(str(rid))
                content = getattr(r, "content", None)
                if isinstance(content, str):
                    contents.append(content)
            used = getattr(response, "strategy_used", None)
            if isinstance(used, str):
                strategy_used = used
        except Exception as exc:
            err = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000.0
        budget = item.latency_budget_ms
        within = err is None and (budget is None or latency_ms <= budget)

        recall_score = recall_at_k(retrieved_ids, item.expected_memory_ids, item.top_k)
        mrr_score = mrr(retrieved_ids, item.expected_memory_ids)
        joined_context = "\n".join(contents)
        entity_score = entity_coverage(joined_context, item.expected_entities)
        verdict = (
            self._judge.judge(
                query=item.query,
                expected_answer=item.expected_answer or "",
                recalled_context=joined_context,
            )
            if item.expected_answer
            else JudgeVerdict(score=math.nan, rationale="no expected_answer")
        )

        return ItemResult(
            item_id=item.id,
            strategy_used=strategy_used,
            retrieved_ids=tuple(retrieved_ids),
            latency_ms=round(latency_ms, 2),
            recall_at_k_score=recall_score,
            mrr_score=mrr_score,
            entity_coverage_score=entity_score,
            faithfulness=verdict,
            within_latency_budget=within,
            error=err,
        )

    def _iter_results(self, response: Any) -> list[Any]:
        """Coerce the SDK response into an iterable of result objects.

        The SDK's ``recall()`` returns a ``RecallResponse`` that is
        directly iterable (yields ``StrategyResult``). Older SDK
        versions returned a list. Both are handled.
        """
        if response is None:
            return []
        results_attr = getattr(response, "results", None)
        if results_attr is not None:
            return list(results_attr)
        try:
            return list(response)
        except TypeError:
            return []
