"""z3rno-evals — eval framework for Z3rno (Phase E).

Public surface:

    from z3rno_evals import (
        EvalRunner,
        GoldenDataset, GoldenItem,
        ItemResult, RunResult,
        recall_at_k, mrr, latency_percentiles,
        FaithfulnessJudge, StubJudge,
    )

Import this package only — sub-modules are implementation detail
unless you're writing tests or extending the judge surface.
"""

from __future__ import annotations

from z3rno_evals.dataset import GoldenDataset, GoldenItem, load_dataset
from z3rno_evals.judges.base import FaithfulnessJudge, JudgeVerdict
from z3rno_evals.judges.stub import StubJudge
from z3rno_evals.metrics import latency_percentiles, mrr, recall_at_k
from z3rno_evals.report import render_json, render_markdown
from z3rno_evals.runner import EvalRunner, ItemResult, RunResult

__version__ = "0.1.0"

__all__ = [
    "EvalRunner",
    "FaithfulnessJudge",
    "GoldenDataset",
    "GoldenItem",
    "ItemResult",
    "JudgeVerdict",
    "RunResult",
    "StubJudge",
    "__version__",
    "latency_percentiles",
    "load_dataset",
    "mrr",
    "recall_at_k",
    "render_json",
    "render_markdown",
]
