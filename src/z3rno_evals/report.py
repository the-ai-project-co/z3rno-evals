"""Report renderers — JSON + Markdown.

JSON is the canonical artifact for the CI regression gate; the
Markdown is for PR-comment display.
"""

from __future__ import annotations

import json
import math
from typing import Any

from z3rno_evals.runner import ItemResult, RunResult


def _fmt(v: float, fmt: str = ".3f") -> str:
    if math.isnan(v):
        return "—"
    return f"{v:{fmt}}"


def render_json(result: RunResult) -> str:
    """Serialize a RunResult to deterministic JSON.

    Field order is stable so two runs of the same dataset diff cleanly.
    """
    payload: dict[str, Any] = {
        "dataset_name": result.dataset_name,
        "dataset_version": result.dataset_version,
        "server_url": result.server_url,
        "summary": {
            "items_total": result.items_total,
            "items_within_budget": result.items_within_budget,
            "mean_recall_at_k": _nan_to_none(result.mean_recall_at_k),
            "mean_mrr": _nan_to_none(result.mean_mrr),
            "mean_entity_coverage": _nan_to_none(result.mean_entity_coverage),
            "mean_faithfulness": _nan_to_none(result.mean_faithfulness),
            "latency": {
                "p50_ms": result.latency.p50,
                "p95_ms": result.latency.p95,
                "p99_ms": result.latency.p99,
                "mean_ms": result.latency.mean,
                "count": result.latency.count,
            },
        },
        "items": [_item_dict(i) for i in result.items],
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def render_markdown(result: RunResult, *, baseline: dict[str, Any] | None = None) -> str:
    """Render a Markdown report — primary surface for PR comments.

    When ``baseline`` is supplied (the prior run's JSON), the summary
    row shows deltas and flags any metric that dropped > 5%.
    """
    lines: list[str] = []
    lines.append(f"# Z3rno evals — `{result.dataset_name}` v{result.dataset_version}")
    lines.append("")
    lines.append(f"Server: `{result.server_url}` · Items: **{result.items_total}**")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value | Δ vs baseline |")
    lines.append("|---|---|---|")
    rows: list[tuple[str, float, str]] = [
        ("recall@k", result.mean_recall_at_k, "mean_recall_at_k"),
        ("MRR", result.mean_mrr, "mean_mrr"),
        ("entity coverage", result.mean_entity_coverage, "mean_entity_coverage"),
        ("faithfulness", result.mean_faithfulness, "mean_faithfulness"),
    ]
    regressions: list[str] = []
    for label, current, key in rows:
        delta_str = _delta(baseline, key, current, regressions, label)
        lines.append(f"| {label} | {_fmt(current)} | {delta_str} |")
    lines.append(f"| p50 latency (ms) | {_fmt(result.latency.p50, '.1f')} | — |")
    lines.append(f"| p95 latency (ms) | {_fmt(result.latency.p95, '.1f')} | — |")
    lines.append(f"| p99 latency (ms) | {_fmt(result.latency.p99, '.1f')} | — |")
    lines.append(f"| items within budget | {result.items_within_budget}/{result.items_total} | — |")
    lines.append("")

    if regressions:
        lines.append("> ⚠ **Regression detected** — the following metrics dropped > 5%:")
        lines.extend(f"> - {r}" for r in regressions)
        lines.append("")

    lines.append("## Per-item results")
    lines.append("")
    lines.append(
        "| ID | Strategy | Latency (ms) | recall@k | MRR | Entity cov. | Faithfulness | Within budget | Error |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.extend(
        "| "
        + " | ".join(
            [
                item.item_id,
                item.strategy_used,
                f"{item.latency_ms:.1f}",
                _fmt(item.recall_at_k_score),
                _fmt(item.mrr_score),
                _fmt(item.entity_coverage_score),
                _fmt(item.faithfulness.score),
                "✅" if item.within_latency_budget else "❌",
                item.error or "—",
            ]
        )
        + " |"
        for item in result.items
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nan_to_none(v: float) -> float | None:
    return None if math.isnan(v) else v


def _item_dict(item: ItemResult) -> dict[str, Any]:
    return {
        "id": item.item_id,
        "strategy_used": item.strategy_used,
        "retrieved_ids": list(item.retrieved_ids),
        "latency_ms": item.latency_ms,
        "recall_at_k": _nan_to_none(item.recall_at_k_score),
        "mrr": _nan_to_none(item.mrr_score),
        "entity_coverage": _nan_to_none(item.entity_coverage_score),
        "faithfulness": {
            "score": _nan_to_none(item.faithfulness.score),
            "rationale": item.faithfulness.rationale,
        },
        "within_latency_budget": item.within_latency_budget,
        "error": item.error,
    }


def _delta(
    baseline: dict[str, Any] | None,
    key: str,
    current: float,
    regressions: list[str],
    label: str,
) -> str:
    """Render the Δ-vs-baseline cell and append to regressions if > 5%."""
    if baseline is None or math.isnan(current):
        return "—"
    prior = (baseline.get("summary") or {}).get(key)
    if prior is None or not isinstance(prior, (int, float)):
        return "—"
    delta = current - prior
    pct = (delta / prior * 100.0) if prior else 0.0
    if pct < -5.0:
        regressions.append(f"{label}: {prior:.3f} → {current:.3f} ({pct:+.1f}%)")
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "·")
    return f"{arrow} {delta:+.3f} ({pct:+.1f}%)"
