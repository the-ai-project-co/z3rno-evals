"""StubJudge — deterministic faithfulness judge for CI and tests.

Scores ``1.0`` when every word of the expected answer (after a
trivial split + lowercase) appears in the recalled context. ``0.0``
otherwise. Crude but reproducible — good enough for regression
detection without an LLM bill.

Skipped (``score=nan``) when ``expected_answer`` is empty.
"""

from __future__ import annotations

import math
import re

from z3rno_evals.judges.base import FaithfulnessJudge, JudgeVerdict

_PUNCT_RE = re.compile(r"[^\w\s]")


class StubJudge(FaithfulnessJudge):
    """Deterministic word-coverage judge. No I/O."""

    def judge(
        self,
        *,
        query: str,
        expected_answer: str,
        recalled_context: str,
    ) -> JudgeVerdict:
        if not expected_answer or not expected_answer.strip():
            return JudgeVerdict(score=math.nan, rationale="no expected_answer")

        # Trivial tokenization. Stop-word filter intentionally minimal —
        # we want regression sensitivity, not perfect linguistics.
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "in",
            "on",
            "at",
            "and",
            "or",
            "for",
        }
        # Strip punctuation before tokenising so "mode." matches "mode".
        clean_expected = _PUNCT_RE.sub(" ", expected_answer.casefold())
        expected_tokens = {w for w in clean_expected.split() if w and w not in stop}
        if not expected_tokens:
            return JudgeVerdict(
                score=math.nan, rationale="expected_answer empty after stop-word filter"
            )

        ctx = _PUNCT_RE.sub(" ", recalled_context.casefold())
        hits = sum(1 for t in expected_tokens if t in ctx)
        score = hits / len(expected_tokens)
        return JudgeVerdict(
            score=score,
            rationale=f"{hits}/{len(expected_tokens)} expected tokens present in recall",
        )
