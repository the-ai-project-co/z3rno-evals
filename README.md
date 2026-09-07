# z3rno-evals

> ## ⚠️ Deprecated — full rewrite in progress
>
> z3rno is being rewritten from the ground up as a single open-source Rust monorepo (Apache 2.0), following direct feedback from enterprise users that the current architecture is too complex.
>
> - **Maintenance mode only** — bug fixes, no new features.
> - Existing installs keep working. Nothing here is being yanked, archived, or removed.
> - The new monorepo lands at `the-ai-project-co/z3rno`.
> - Track progress: https://github.com/the-ai-project-co/z3rno-evals/issues/3

Eval framework for [Z3rno](https://z3rno.dev). Measures the four metrics that gate every Phase E release:

- **recall@k** — fraction of golden questions where an expected Memo is in the top-k results
- **MRR** — mean reciprocal rank of the first correct Memo
- **faithfulness** — LLM-judge score: does the recalled context support the expected answer?
- **latency** — p50 / p95 / p99 of `recall()` calls against a live z3rno-server

A golden dataset is a JSON file of Q&A items with expected memory IDs, expected entities, and latency budgets. The CLI runs the dataset against a live server and emits both a `results.json` (machine-readable) and a `report.md` (PR-comment-ready).

## Install

```bash
pip install z3rno-evals
# Optional — for the LLM-driven faithfulness judge
pip install 'z3rno-evals[llm-judge]'
```

## Quickstart

```bash
# 1. Boot a local z3rno-server (see z3rno-server/Makefile)
make -C ../z3rno-server dev-up

# 2. Seed the memories the golden dataset expects
python -m z3rno_evals.seed --against http://localhost:8000 --api-key sk-... \
    --dataset datasets/golden_v1.json

# 3. Run the evals
z3rno-evals run \
    --against http://localhost:8000 \
    --api-key sk-... \
    --dataset datasets/golden_v1.json \
    --output results/

# 4. Look at the report
cat results/report.md
```

## Metrics Table

| Metric | Default Target | Phase E Bar |
|---|---|---|
| `recall@5` (`GRAPH` strategy) | — | ≥ 0.80 on golden v1 |
| `mrr` | — | ≥ 0.65 |
| `faithfulness` (LLM-judge) | — | ≥ 0.90 |
| `p95_latency_ms` | — | ≤ 500 |
| Regression gate | — | block PR if any metric drops > 5% |

## Golden Dataset Format

```json
{
  "version": 1,
  "name": "golden_v1",
  "items": [
    {
      "id": "q-001",
      "agent_id": "agent-1",
      "query": "What does the user prefer for notifications?",
      "expected_memory_ids": ["01HXY1...", "01HXY2..."],
      "expected_entities": ["dark mode", "weekly digest"],
      "expected_answer": "The user prefers dark mode and weekly digest emails.",
      "strategy": "AUTO",
      "top_k": 5,
      "latency_budget_ms": 500
    }
  ]
}
```

See [`datasets/golden_v1.json`](datasets/golden_v1.json) for the full sample.

## Architecture

- `src/z3rno_evals/dataset.py` — Pydantic models + loader for the golden dataset format
- `src/z3rno_evals/metrics.py` — pure-function metric implementations
- `src/z3rno_evals/runner.py` — drives the eval against a live server via the `z3rno` SDK
- `src/z3rno_evals/judges/` — `FaithfulnessJudge` ABC + LLM + stub implementations
- `src/z3rno_evals/report.py` — JSON + Markdown report renderers
- `src/z3rno_evals/cli.py` — argparse entry point (`z3rno-evals run`, `z3rno-evals seed`)

## License

Apache 2.0.
