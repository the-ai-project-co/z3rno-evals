"""Unit tests for the StubJudge (LLMJudge is mocked separately)."""

from __future__ import annotations

import math

from z3rno_evals.judges.stub import StubJudge


def test_stub_judge_full_coverage() -> None:
    verdict = StubJudge().judge(
        query="What does the user prefer?",
        expected_answer="The user prefers dark mode.",
        recalled_context="user prefers dark mode",
    )
    assert verdict.score == 1.0


def test_stub_judge_partial_coverage() -> None:
    verdict = StubJudge().judge(
        query="What does the user prefer?",
        expected_answer="The user prefers dark mode and weekly digests.",
        recalled_context="user prefers dark mode",
    )
    assert 0.0 < verdict.score < 1.0


def test_stub_judge_zero_coverage() -> None:
    verdict = StubJudge().judge(
        query="What?",
        expected_answer="dark mode preferences",
        recalled_context="completely unrelated content",
    )
    assert verdict.score == 0.0


def test_stub_judge_empty_expected_is_nan() -> None:
    verdict = StubJudge().judge(query="q", expected_answer="", recalled_context="ctx")
    assert math.isnan(verdict.score)


def test_stub_judge_only_stop_words_is_nan() -> None:
    verdict = StubJudge().judge(query="q", expected_answer="the a an", recalled_context="ctx")
    assert math.isnan(verdict.score)
