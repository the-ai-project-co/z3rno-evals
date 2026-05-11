"""Unit tests for the pure metric functions."""

from __future__ import annotations

import math

from z3rno_evals.metrics import (
    content_recall_at_k,
    entity_coverage,
    latency_percentiles,
    mrr,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------


def test_recall_at_k_all_hit() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "b"], 5) == 1.0


def test_recall_at_k_partial_hit() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "x"], 5) == 0.5


def test_recall_at_k_zero_hit() -> None:
    assert recall_at_k(["a", "b", "c"], ["x", "y"], 5) == 0.0


def test_recall_at_k_respects_k() -> None:
    """An expected id outside the top-k window doesn't count."""
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0


def test_recall_at_k_empty_expected_is_nan() -> None:
    assert math.isnan(recall_at_k(["a"], [], 5))


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


def test_mrr_first_position() -> None:
    assert mrr(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_third_position() -> None:
    assert mrr(["x", "y", "a"], ["a"]) == 1 / 3


def test_mrr_no_match_returns_zero() -> None:
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_mrr_empty_expected_is_nan() -> None:
    assert math.isnan(mrr(["a"], []))


# ---------------------------------------------------------------------------
# latency_percentiles
# ---------------------------------------------------------------------------


def test_latency_percentiles_empty() -> None:
    p = latency_percentiles([])
    assert p.count == 0
    assert p.p50 == 0.0
    assert p.p95 == 0.0


def test_latency_percentiles_basic() -> None:
    p = latency_percentiles([100.0, 200.0, 300.0, 400.0, 500.0])
    assert p.count == 5
    assert p.p50 == 300.0
    assert p.p95 == 500.0
    assert p.p99 == 500.0
    assert p.mean == 300.0


def test_latency_percentiles_p95_picks_correct_index() -> None:
    """With 100 values 1..100, p95 should be 95."""
    p = latency_percentiles([float(i) for i in range(1, 101)])
    assert p.p95 == 95.0
    assert p.p99 == 99.0
    assert p.p50 == 50.0


# ---------------------------------------------------------------------------
# entity_coverage
# ---------------------------------------------------------------------------


def test_entity_coverage_all_hit() -> None:
    assert entity_coverage("Ada Lovelace works at Anthropic.", ["Ada", "Anthropic"]) == 1.0


def test_entity_coverage_partial() -> None:
    assert entity_coverage("Ada Lovelace works at Google.", ["Ada", "Anthropic"]) == 0.5


def test_entity_coverage_is_case_insensitive() -> None:
    assert entity_coverage("ada lovelace", ["ADA"]) == 1.0


def test_entity_coverage_empty_expected_is_nan() -> None:
    assert math.isnan(entity_coverage("anything", []))


# ---------------------------------------------------------------------------
# content_recall_at_k — UUID-free recall
# ---------------------------------------------------------------------------


def test_content_recall_full_hit() -> None:
    contents = ["Ada Lovelace works at Anthropic.", "noise"]
    assert content_recall_at_k(contents, ["Ada", "Anthropic"], 5) == 1.0


def test_content_recall_partial_hit() -> None:
    contents = ["Ada Lovelace works at Google."]
    assert content_recall_at_k(contents, ["Ada", "Anthropic"], 5) == 0.5


def test_content_recall_respects_top_k() -> None:
    contents = ["unrelated", "unrelated", "Ada"]
    # k=2 excludes the third row → no hit
    assert content_recall_at_k(contents, ["Ada"], 2) == 0.0
    assert content_recall_at_k(contents, ["Ada"], 3) == 1.0


def test_content_recall_is_case_insensitive() -> None:
    assert content_recall_at_k(["ADA LOVELACE"], ["ada"], 5) == 1.0


def test_content_recall_empty_expected_is_nan() -> None:
    assert math.isnan(content_recall_at_k(["x"], [], 5))
