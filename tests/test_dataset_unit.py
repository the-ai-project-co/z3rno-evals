"""Unit tests for the golden dataset schema + loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from z3rno_evals.dataset import GoldenDataset, GoldenItem, load_dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_V1 = _REPO_ROOT / "datasets" / "golden_v1.json"


def test_load_shipped_golden_v1() -> None:
    """The dataset we ship in repo must parse cleanly."""
    dataset = load_dataset(_GOLDEN_V1)
    assert dataset.version == 1
    assert dataset.name == "golden_v1"
    assert len(dataset.items) >= 10


def test_golden_item_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        GoldenItem(id="x", agent_id="a", query="")


def test_golden_item_rejects_oversize_top_k() -> None:
    with pytest.raises(ValidationError):
        GoldenItem(id="x", agent_id="a", query="q", top_k=10_000)


def test_golden_dataset_rejects_empty_items() -> None:
    with pytest.raises(ValidationError):
        GoldenDataset(version=1, name="empty", items=())


def test_golden_dataset_by_id() -> None:
    dataset = load_dataset(_GOLDEN_V1)
    item = dataset.by_id("q-001")
    assert item is not None
    assert item.id == "q-001"
    assert dataset.by_id("nope") is None


def test_load_dataset_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/golden.json")


def test_golden_dataset_rejects_extra_fields(tmp_path: Path) -> None:
    """frozen + extra='forbid' guards against schema drift."""
    payload = {
        "version": 1,
        "name": "bad",
        "items": [{"id": "x", "agent_id": "a", "query": "q", "rogue_field": "x"}],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValidationError):
        load_dataset(p)
