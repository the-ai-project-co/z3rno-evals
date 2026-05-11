"""Pure-function metric implementations.

No I/O, no DB, no LLM — testable in isolation. The runner composes
these with judge outputs to produce the final report.
"""

from __future__ import annotations

from dataclasses import dataclass


def recall_at_k(
    retrieved_ids: list[str], expected_ids: list[str] | tuple[str, ...], k: int
) -> float:
    """Fraction of expected IDs that appear in the top-k retrieved.

    Returns 1.0 when every expected ID is hit, 0.0 when none are.
    When ``expected_ids`` is empty, returns ``float("nan")`` — the
    item contributes no recall signal (e.g. aggregate queries where
    expected_memory_ids = []).
    """
    if not expected_ids:
        return float("nan")
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in top_k)
    return hits / len(expected_ids)


def mrr(retrieved_ids: list[str], expected_ids: list[str] | tuple[str, ...]) -> float:
    """Mean reciprocal rank of the first expected ID in the retrieved list.

    Returns ``float("nan")`` when no expected IDs are provided. Returns
    0.0 when none of the expected IDs appear in the retrieved list.
    """
    if not expected_ids:
        return float("nan")
    expected = set(expected_ids)
    for rank, mid in enumerate(retrieved_ids, start=1):
        if mid in expected:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class LatencyPercentiles:
    p50: float
    p95: float
    p99: float
    mean: float
    count: int


def latency_percentiles(latencies_ms: list[float]) -> LatencyPercentiles:
    """Compute p50/p95/p99 over a list of millisecond latencies.

    Empty input → all percentiles 0.0 with count 0. Otherwise uses the
    nearest-rank method (sorted index = ceil(p * n) - 1) — good enough
    for eval reports; we don't need interpolation precision.
    """
    if not latencies_ms:
        return LatencyPercentiles(p50=0.0, p95=0.0, p99=0.0, mean=0.0, count=0)
    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)

    def _pct(p: float) -> float:
        # Nearest-rank: index = ceil(p * n) - 1, clamped to [0, n-1]
        idx = max(0, min(n - 1, int(p * n + 0.999999) - 1 if p * n > 0 else 0))
        return sorted_l[idx]

    return LatencyPercentiles(
        p50=_pct(0.50),
        p95=_pct(0.95),
        p99=_pct(0.99),
        mean=sum(sorted_l) / n,
        count=n,
    )


def entity_coverage(content: str, expected_entities: tuple[str, ...] | list[str]) -> float:
    """Fraction of expected entity strings appearing (case-insensitive substring) in content.

    Useful as a cheap faithfulness pre-filter before invoking the
    LLM judge — if the recalled content doesn't even mention the
    entities, no need to spend a token on the judge.
    """
    if not expected_entities:
        return float("nan")
    lc = content.casefold()
    hits = sum(1 for e in expected_entities if e.casefold() in lc)
    return hits / len(expected_entities)
