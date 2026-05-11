"""Faithfulness judge ABC + verdict shape."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeVerdict:
    """One faithfulness decision."""

    score: float  # 0.0 .. 1.0
    rationale: str = ""


class FaithfulnessJudge(ABC):
    """Score whether ``recalled_context`` supports ``expected_answer``."""

    @abstractmethod
    def judge(
        self,
        *,
        query: str,
        expected_answer: str,
        recalled_context: str,
    ) -> JudgeVerdict:
        """Return a 0..1 faithfulness score with optional rationale."""
