# CLAUDE.md

## Project

z3rno-evals is the eval framework for Z3rno (Phase E). Drives a golden Q&A dataset against a live z3rno-server and reports recall@k, MRR, LLM-judged faithfulness, and latency percentiles. Gates regressions on every PR.

## Quick Reference

```bash
uv sync --dev                              # Install dependencies
uv run ruff check .                        # Lint
uv run ruff format .                       # Format
uv run mypy .                              # Type check (strict)
uv run pytest                              # Run unit tests
uv run z3rno-evals run --against URL ...   # Drive a live eval
```

## Architecture

- `src/z3rno_evals/dataset.py` — Pydantic models + loader for golden JSON.
- `src/z3rno_evals/metrics.py` — pure-function `recall_at_k`, `mrr`, `latency_percentiles`.
- `src/z3rno_evals/judges/` — `FaithfulnessJudge` ABC + `LLMJudge` (litellm) + `StubJudge` (deterministic, for tests + no-LLM env).
- `src/z3rno_evals/runner.py` — `EvalRunner` orchestrates per-item recall + metric computation.
- `src/z3rno_evals/report.py` — emits both `results.json` and `report.md`.
- `src/z3rno_evals/cli.py` — `z3rno-evals run | seed | judge-only`.
- `datasets/golden_v1.json` — ships ~10 hand-crafted items, the baseline regression set.

## Key Conventions

- Python 3.10+, src/ layout, hatchling.
- Zero z3rno-server / z3rno-core imports — talks to the server purely via the public `z3rno` SDK.
- LLM judge is opt-in: `pip install 'z3rno-evals[llm-judge]'`; default judge is the deterministic stub.
- All metric functions are pure (no I/O) so they're trivially testable.
- Conventional commits.

## Phase E Acceptance Bars

| Metric | Bar |
|---|---|
| `recall@5` (GRAPH strategy) | ≥ 0.80 |
| `mrr` | ≥ 0.65 |
| `faithfulness` | ≥ 0.90 |
| `p95_latency_ms` | ≤ 500 |
| Regression gate | block PR if any metric drops > 5% |
