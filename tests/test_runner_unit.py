"""Unit tests for EvalRunner with a fake SDK client.

Verifies the runner: handles successful recalls, captures latency,
computes per-item metrics, aggregates ignoring NaN signals, and
records errors when the client raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from z3rno_evals.dataset import GoldenDataset, GoldenItem
from z3rno_evals.judges.stub import StubJudge
from z3rno_evals.runner import EvalRunner

# ---------------------------------------------------------------------------
# Fake SDK shapes
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    memory_id: str
    content: str


@dataclass
class _FakeResponse:
    results: list[_FakeResult]
    strategy_used: str = "AUTO"


class _FakeClient:
    def __init__(self, responses: dict[str, _FakeResponse], raise_for_item: str | None = None):
        self._responses = responses
        self._raise_for = raise_for_item

    def recall(self, *, agent_id: str, query: str, top_k: int, strategy: str) -> Any:
        if self._raise_for is not None and query == self._raise_for:
            raise RuntimeError("simulated transient failure")
        return self._responses.get(query, _FakeResponse(results=[]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _dataset() -> GoldenDataset:
    return GoldenDataset(
        version=1,
        name="test",
        items=(
            GoldenItem(
                id="i1",
                agent_id="a",
                query="hit",
                expected_memory_ids=("m1",),
                expected_entities=("dark mode",),
                expected_answer="user prefers dark mode",
                top_k=3,
                latency_budget_ms=10_000,
            ),
            GoldenItem(
                id="i2",
                agent_id="a",
                query="miss",
                expected_memory_ids=("m2",),
                top_k=3,
                latency_budget_ms=10_000,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_hit_produces_recall_one() -> None:
    client = _FakeClient(
        responses={
            "hit": _FakeResponse(
                results=[_FakeResult(memory_id="m1", content="user prefers dark mode")]
            ),
            "miss": _FakeResponse(results=[_FakeResult(memory_id="mX", content="unrelated")]),
        }
    )
    runner = EvalRunner(client=client, judge=StubJudge())
    result = runner.run(_dataset(), server_url="http://test")

    by_id = {r.item_id: r for r in result.items}
    assert by_id["i1"].recall_at_k_score == 1.0
    assert by_id["i1"].mrr_score == 1.0
    assert by_id["i1"].faithfulness.score == 1.0
    assert by_id["i2"].recall_at_k_score == 0.0
    assert by_id["i2"].mrr_score == 0.0


def test_runner_aggregates_mean_ignoring_nan() -> None:
    """Items with no expected_memory_ids should not drag the mean down."""
    client = _FakeClient(responses={})
    dataset = GoldenDataset(
        version=1,
        name="t",
        items=(
            GoldenItem(id="i1", agent_id="a", query="q1", expected_memory_ids=("m1",), top_k=3),
            GoldenItem(id="i2", agent_id="a", query="q2", expected_memory_ids=(), top_k=3),
        ),
    )
    result = EvalRunner(client=client).run(dataset, server_url="http://t")
    # i1 contributes 0.0 (miss). i2 contributes NaN (no expectation).
    # Mean over [0.0] = 0.0.
    assert result.mean_recall_at_k == 0.0


def test_runner_records_error_on_client_exception() -> None:
    client = _FakeClient(responses={}, raise_for_item="boom")
    dataset = GoldenDataset(
        version=1,
        name="t",
        items=(GoldenItem(id="i", agent_id="a", query="boom", top_k=3),),
    )
    result = EvalRunner(client=client).run(dataset, server_url="http://t")
    assert result.items[0].error is not None
    assert "simulated" in result.items[0].error
    assert result.items[0].within_latency_budget is False


def test_runner_picks_up_strategy_used_from_response() -> None:
    client = _FakeClient(responses={"hit": _FakeResponse(results=[], strategy_used="VECTOR")})
    dataset = GoldenDataset(
        version=1,
        name="t",
        items=(GoldenItem(id="i", agent_id="a", query="hit", strategy="AUTO", top_k=3),),
    )
    result = EvalRunner(client=client).run(dataset, server_url="http://t")
    assert result.items[0].strategy_used == "VECTOR"


def test_runner_latency_percentiles_have_count() -> None:
    client = _FakeClient(responses={})
    dataset = GoldenDataset(
        version=1,
        name="t",
        items=(
            GoldenItem(id="i1", agent_id="a", query="q", top_k=3),
            GoldenItem(id="i2", agent_id="a", query="q", top_k=3),
        ),
    )
    result = EvalRunner(client=client).run(dataset, server_url="http://t")
    assert result.latency.count == 2
    assert not math.isnan(result.latency.p50)
