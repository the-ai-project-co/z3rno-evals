"""Golden dataset schema + loader.

A golden dataset is a versioned JSON file of Q&A items. Each item
records an expected recall (which memory IDs the harness should
surface) plus optional faithfulness checks (entity strings the
recalled context must mention, expected free-text answer).

Frozen Pydantic models — datasets are checked into source control and
should never mutate at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DatasetVersion = Literal[1]


class GoldenItem(BaseModel):
    """One Q&A item in a golden dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1, description="Stable identifier within the dataset.")
    agent_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    expected_memory_ids: tuple[str, ...] = Field(
        default=(),
        description="Memo ids the recall MUST surface. Empty → recall metrics skipped for this item.",
    )
    expected_entities: tuple[str, ...] = Field(
        default=(),
        description="Substrings the recalled context must contain. Empty → entity check skipped.",
    )
    expected_answer: str | None = Field(
        default=None,
        description="Free-text answer for the faithfulness judge.",
    )
    strategy: str = Field(default="AUTO")
    top_k: int = Field(default=5, ge=1, le=100)
    latency_budget_ms: int | None = Field(
        default=None,
        ge=1,
        description="Per-item override; falls back to the run-wide default.",
    )
    tags: tuple[str, ...] = Field(
        default=(),
        description="Free-form labels (e.g. 'phase-d', 'graph-strategy'). Used for grouped report rows.",
    )


class GoldenDataset(BaseModel):
    """Top-level container for a versioned golden dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: DatasetVersion
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    items: tuple[GoldenItem, ...] = Field(..., min_length=1)

    def by_id(self, item_id: str) -> GoldenItem | None:
        for item in self.items:
            if item.id == item_id:
                return item
        return None


def load_dataset(path: str | Path) -> GoldenDataset:
    """Load + validate a golden dataset from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"golden dataset not found: {p}")
    payload = json.loads(p.read_text())
    return GoldenDataset.model_validate(payload)
