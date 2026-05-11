"""Unit tests for the report renderer (JSON + Markdown)."""

from __future__ import annotations

import json

from z3rno_evals.judges.base import JudgeVerdict
from z3rno_evals.metrics import LatencyPercentiles
from z3rno_evals.report import render_json, render_markdown
from z3rno_evals.runner import ItemResult, RunResult


def _run_result(**overrides: object) -> RunResult:
    defaults: dict[str, object] = {
        "dataset_name": "test",
        "dataset_version": 1,
        "server_url": "http://test",
        "items": (
            ItemResult(
                item_id="i1",
                strategy_used="AUTO",
                retrieved_ids=("m1",),
                latency_ms=42.0,
                recall_at_k_score=1.0,
                mrr_score=1.0,
                entity_coverage_score=1.0,
                faithfulness=JudgeVerdict(score=0.9, rationale="ok"),
                within_latency_budget=True,
            ),
        ),
        "latency": LatencyPercentiles(p50=42.0, p95=42.0, p99=42.0, mean=42.0, count=1),
        "mean_recall_at_k": 1.0,
        "mean_mrr": 1.0,
        "mean_entity_coverage": 1.0,
        "mean_faithfulness": 0.9,
        "items_within_budget": 1,
        "items_total": 1,
    }
    defaults.update(overrides)
    return RunResult(**defaults)  # type: ignore[arg-type]


def test_render_json_is_valid_json() -> None:
    payload = json.loads(render_json(_run_result()))
    assert payload["dataset_name"] == "test"
    assert payload["summary"]["mean_recall_at_k"] == 1.0
    assert payload["items"][0]["id"] == "i1"


def test_render_markdown_contains_summary_table() -> None:
    md = render_markdown(_run_result())
    assert "# Z3rno evals" in md
    assert "## Summary" in md
    assert "## Per-item results" in md
    assert "recall@k" in md


def test_render_markdown_no_baseline_shows_em_dash() -> None:
    md = render_markdown(_run_result())
    assert "Δ vs baseline" in md
    # Every delta cell should be em-dash when no baseline.
    summary_section = md.split("## Per-item")[0]
    assert summary_section.count("| — |") >= 4


def test_render_markdown_flags_regression() -> None:
    baseline = {
        "summary": {
            "mean_recall_at_k": 0.90,  # current is 1.0 — improvement
            "mean_mrr": 1.0,
            "mean_entity_coverage": 1.0,
            "mean_faithfulness": 1.0,  # current 0.9 = 10% drop → regression
        }
    }
    md = render_markdown(_run_result(), baseline=baseline)
    assert "**Regression detected**" in md
    assert "faithfulness" in md
