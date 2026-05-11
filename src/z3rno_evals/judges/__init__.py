"""Faithfulness judges — pluggable scorers that decide whether the
recalled context supports the expected answer.

Two implementations ship:

  * :class:`StubJudge` — deterministic, no LLM. Returns 1.0 when
    every expected entity appears in the recalled context, 0.0
    otherwise. Default in CI when no LLM key is available.
  * :class:`LLMJudge` — talks to a real LLM via litellm. Opt-in via
    the ``[llm-judge]`` extra.
"""

from __future__ import annotations

from z3rno_evals.judges.base import FaithfulnessJudge, JudgeVerdict
from z3rno_evals.judges.stub import StubJudge

__all__ = ["FaithfulnessJudge", "JudgeVerdict", "StubJudge"]
