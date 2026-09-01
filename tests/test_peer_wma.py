"""Release-gate checks for the two-Claude-session WMA study path."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def _validator():
    path = REPO / "rollout" / "validate_wma_session.py"
    spec = importlib.util.spec_from_file_location("peer_wma_validator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _session(tmp_path: Path, condition: str = "c2") -> tuple[Path, dict]:
    task = tmp_path / "task"
    wm = task / "wm"
    card_dir = wm / "cards" / "exp-01"
    card_dir.mkdir(parents=True)
    (task / "final_model").mkdir()
    (task / "final_model" / "config.json").write_text("{}\n")

    arm = "traj" if condition == "c2" else "retrieval"
    config = {
        "schema_version": "awm-wm-config-v2",
        "session_id": "task-1",
        "session_dir": str(task),
        "arm": arm,
        "prior_runs_root": "/home/ben/prior_runs" if condition == "c2" else None,
        "memory_root": "/home/ben/wm-memory" if condition == "c3" else None,
        "memory_sides": ["train"],
        "memory_readonly": True,
        "split_side": "test",
        "wma_model": "claude-opus-4-8@fixed",
        "base_model": "google/gemma-3-4b-pt",
        "consult_api": "SendMessage to the wma session; awm-consult-response-v1",
    }
    (wm / "config.json").write_text(json.dumps(config))
    card = {
        "schema_version": "awm-experiment-card-v1",
        "card_id": "exp-01",
        "problem": {},
        "hypothesis": {},
        "setup": {},
        "evaluation": {},
    }
    (card_dir / "card.json").write_text(json.dumps(card))
    response = card_dir / "consult-01.json"
    response.write_text(json.dumps({"request": "plan", "response": {"card": card}}))
    rows = [
        {
            "seq": 1,
            "card_id": "exp-01",
            "arm": arm,
            "model": "claude-opus-4-8@fixed",
            "verdict": "CANNOT_DECIDE",
            "suggestion": "KEEP_RUNNING",
            "path": str(response),
        },
        {
            "seq": 2,
            "event": "outcome",
            "card_id": "exp-01",
            "final_value": 0.5,
            "shipped": "/home/ben/task/final_model",
        },
    ]
    (wm / "consults.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (wm / "wma-session.jsonl").write_text(
        json.dumps({"type": "system", "subtype": "init", "model": "claude-opus-4-8@fixed"})
        + "\n"
        + json.dumps({"type": "result", "subtype": "success"})
        + "\n"
    )
    (wm / "wma-exit-code.txt").write_text("0\n")
    (wm / "wma-capture-exit-code.txt").write_text("0\n")
    study = {
        "condition": condition,
        "wma_model": {
            "expected_model_id": "claude-opus-4-8@fixed",
            "reported_model_ids": ["claude-opus-4-8@fixed"],
            "reported_providers": ["vertex"],
        },
    }
    return task, study


@pytest.mark.parametrize("condition", ("c2", "c3"))
def test_peer_session_attests_consult_and_shipped_outcome(
    tmp_path: Path, condition: str
) -> None:
    validator = _validator()
    task, study = _session(tmp_path, condition)
    evidence = validator.validate_peer_session(
        task,
        expected_arm="traj" if condition == "c2" else "retrieval",
        expected_wma_model="claude-opus-4-8@fixed",
        expected_memory_sides="train",
        study=study,
    )
    assert evidence["protocol"] == "peer-consult-v1"
    assert evidence["consult_count"] == 1
    assert evidence["outcome_count"] == 1
    assert evidence["card_ids"] == ["exp-01"]


def test_peer_session_rejects_a_labelled_cell_without_outcome(tmp_path: Path) -> None:
    validator = _validator()
    task, study = _session(tmp_path)
    ledger = task / "wm" / "consults.jsonl"
    ledger.write_text(ledger.read_text().splitlines()[0] + "\n")
    with pytest.raises(validator.ValidationError, match="what it shipped"):
        validator.validate_peer_session(
            task,
            expected_arm="traj",
            expected_wma_model="claude-opus-4-8@fixed",
            expected_memory_sides="train",
            study=study,
        )
