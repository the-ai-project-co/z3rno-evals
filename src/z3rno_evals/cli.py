"""``z3rno-evals`` CLI entry point.

Two subcommands:

  * ``z3rno-evals run`` — execute a golden dataset against a live server,
    write ``results.json`` + ``report.md``, and exit non-zero on
    regression (> 5% drop vs ``--baseline``).
  * ``z3rno-evals seed`` — populate a fresh server with the memories the
    dataset expects. Useful for CI sandboxes; the JSON payload lives
    next to the dataset under ``<name>.seed.json`` if present.

The CLI is intentionally minimal — argparse, no fancy progress bars,
no colors. Output is grep-able for CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from z3rno_evals.dataset import load_dataset
from z3rno_evals.judges.stub import StubJudge
from z3rno_evals.report import render_json, render_markdown
from z3rno_evals.runner import EvalRunner

_EXIT_OK = 0
_EXIT_RUN_FAILED = 1
_EXIT_REGRESSION = 2


def _make_client(base_url: str, api_key: str) -> Any:
    """Construct a z3rno.Z3rnoClient. Lazy-imported so unit tests can run
    without the SDK installed."""
    try:
        from z3rno import Z3rnoClient  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "the 'z3rno' SDK is required at runtime. Install with: pip install z3rno"
        ) from exc
    return Z3rnoClient(api_key=api_key, base_url=base_url)


def _make_judge(args: argparse.Namespace) -> Any:
    if args.judge == "stub":
        return StubJudge()
    if args.judge == "llm":
        from z3rno_evals.judges.llm import LLMJudge  # noqa: PLC0415

        return LLMJudge(
            model=args.judge_model,
            api_key=args.judge_api_key or os.environ.get("OPENAI_API_KEY", ""),
        )
    raise ValueError(f"unknown judge: {args.judge}")


def _cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline: dict[str, Any] | None = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
        else:
            print(
                f"warning: baseline not found at {baseline_path}; running without baseline",
                file=sys.stderr,
            )

    client = _make_client(args.against, args.api_key)
    judge = _make_judge(args)
    runner = EvalRunner(client=client, judge=judge)

    print(f"running {dataset.name} ({len(dataset.items)} items) against {args.against}...")
    try:
        result = runner.run(dataset, server_url=args.against)
    except Exception as exc:
        print(f"error: eval run failed: {exc}", file=sys.stderr)
        return _EXIT_RUN_FAILED

    json_path = output_dir / "results.json"
    md_path = output_dir / "report.md"
    json_path.write_text(render_json(result))
    md_path.write_text(render_markdown(result, baseline=baseline))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")

    # Regression gate — surface the same warning the markdown already
    # contains so CI exits non-zero when any metric drops > 5%.
    rendered = md_path.read_text()
    if "**Regression detected**" in rendered:
        print("regression detected — see report.md", file=sys.stderr)
        return _EXIT_REGRESSION

    return _EXIT_OK


def _cmd_seed(args: argparse.Namespace) -> int:
    """Naïvely seed expected memories from a ``<name>.seed.json`` file.

    The seed file is a list of ``store()`` payloads. Missing → exit
    cleanly with a hint (the harness can still run against a
    pre-seeded server).
    """
    seed_path = Path(args.seed) if args.seed else Path(args.dataset).with_suffix(".seed.json")
    if not seed_path.exists():
        print(f"no seed file at {seed_path}; nothing to do.")
        return _EXIT_OK
    payloads: list[dict[str, Any]] = json.loads(seed_path.read_text())
    client = _make_client(args.against, args.api_key)
    print(f"seeding {len(payloads)} memories from {seed_path}...")
    for p in payloads:
        client.store(**p)
    print("seed complete.")
    return _EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="z3rno-evals", description="Z3rno eval harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- run ---
    run = sub.add_parser("run", help="execute a golden dataset against a live server")
    run.add_argument("--against", required=True, help="base URL of z3rno-server")
    run.add_argument("--api-key", default=os.environ.get("Z3RNO_API_KEY", ""))
    run.add_argument("--dataset", required=True, help="path to a golden_*.json file")
    run.add_argument("--output", default="results", help="output directory (default: ./results)")
    run.add_argument("--baseline", default="", help="prior results.json for regression-delta")
    run.add_argument("--judge", choices=["stub", "llm"], default="stub")
    run.add_argument("--judge-model", default="openai/gpt-4o-mini")
    run.add_argument("--judge-api-key", default="")
    run.set_defaults(func=_cmd_run)

    # --- seed ---
    seed = sub.add_parser("seed", help="seed memories from a <dataset>.seed.json file")
    seed.add_argument("--against", required=True)
    seed.add_argument("--api-key", default=os.environ.get("Z3RNO_API_KEY", ""))
    seed.add_argument("--dataset", required=True)
    seed.add_argument(
        "--seed", default="", help="explicit seed file path (default: derived from --dataset)"
    )
    seed.set_defaults(func=_cmd_seed)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
