"""LLMJudge — litellm-backed faithfulness scorer.

Opt-in via the ``[llm-judge]`` extra. Lazy-imports litellm so the
default install stays light. The judge asks an LLM to score in
[0, 1] whether the recalled context entails the expected answer.

A single retry is attempted on parse failure; the second failure
returns ``score=nan`` with a "judge_failed" rationale rather than
raising — eval runs should be resilient to flaky LLMs.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from z3rno_evals.judges.base import FaithfulnessJudge, JudgeVerdict

log = logging.getLogger(__name__)


_SYSTEM = (
    "You are an evaluation judge. Score whether the recalled context supports the "
    "expected answer to the given query. Respond with a single JSON object: "
    '{"score": <float 0..1>, "rationale": "<one sentence>"}. '
    "1.0 means the context fully supports the answer; 0.0 means it doesn't. "
    "No prose outside the JSON."
)


def _build_user_prompt(query: str, expected_answer: str, recalled_context: str) -> str:
    return (
        f"Query: {query}\n\n"
        f"Expected answer: {expected_answer}\n\n"
        f"Recalled context:\n{recalled_context}\n"
    )


def _parse_verdict(raw: str) -> JudgeVerdict | None:
    # Extract the first {...} block; LLMs sometimes wrap JSON in prose.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        payload: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    score = payload.get("score")
    if not isinstance(score, (int, float)):
        return None
    score = max(0.0, min(1.0, float(score)))
    rationale = str(payload.get("rationale", ""))[:500]
    return JudgeVerdict(score=score, rationale=rationale)


class LLMJudge(FaithfulnessJudge):
    """Faithfulness judge driven by an LLM via litellm."""

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o-mini",
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def judge(
        self,
        *,
        query: str,
        expected_answer: str,
        recalled_context: str,
    ) -> JudgeVerdict:
        if not expected_answer or not expected_answer.strip():
            return JudgeVerdict(score=math.nan, rationale="no expected_answer")

        try:
            import litellm  # noqa: PLC0415 — lazy
        except ImportError as exc:
            raise ImportError(
                "litellm is required for the LLM judge. "
                "Install with: pip install 'z3rno-evals[llm-judge]'"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": _build_user_prompt(query, expected_answer, recalled_context),
                },
            ],
            "temperature": 0.0,
            "timeout": self.timeout,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key

        for _ in range(2):
            try:
                response = litellm.completion(**kwargs)
                raw = response["choices"][0]["message"]["content"]
            except Exception as exc:
                log.warning("llm_judge.completion_failed", exc_info=exc)
                continue
            parsed = _parse_verdict(raw)
            if parsed is not None:
                return parsed
        return JudgeVerdict(score=math.nan, rationale="judge_failed")
