import copy
import hashlib
import json

import pytest

from tools.outcome_prediction import wm_dataset, wm_prepare
from tools.outcome_prediction.wm_model import RecipeWorldModel


def test_preparation_freezes_whole_runs_and_separates_test_labels(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    rows = []
    for cell in ("r0-01", "r0-02", "r0-03", "r0-04"):
        folder = raw_root / "cells" / cell
        folder.mkdir(parents=True)
        (folder / "solve_out_sanitized.txt").write_text(f"raw trajectory for {cell}\n")
        for number in range(1, 4):
            rows.append(
                {
                    "example_id": f"{cell}/exp-{number:02d}",
                    "cell_id": cell,
                    "benchmark": "gsm8k",
                    "scientist_model": "same-scientist",
                    "model_input": {
                        "task": {"benchmark": "gsm8k"},
                        "plan": {"setup": {"lr": 1e-5 * number}},
                    },
                    "label": {"accuracy": number / 10},
                    "audit": {"eligible": True},
                }
            )
    (raw_root / "manifest.json").write_text(
        json.dumps([{"cell_id": c} for c in sorted({r["cell_id"] for r in rows})])
    )
    monkeypatch.setattr(
        wm_dataset, "extract_recorder_dataset", lambda **kwargs: (copy.deepcopy(rows), {})
    )
    source = {
        "raw_root": str(raw_root),
        "manifest_name": "manifest.json",
        "source_revision": "pinned",
        "benchmark": "gsm8k",
        "base_model": "base",
        "evaluation_n": 1319,
    }
    output = tmp_path / "prepared"
    protocol = wm_prepare.prepare(sources=[source], output_dir=output, test_count=1, seed=123)
    split = json.loads((output / "split.json").read_text())
    assert len(split["train_cell_ids"]) == 3
    assert len(split["test_cell_ids"]) == 1
    assert set(split["train_cell_ids"]).isdisjoint(split["test_cell_ids"])
    model = RecipeWorldModel.load(
        output / "model/wm.joblib", expected_sha256=protocol["model_sha256"]
    )
    assert set(model.training_manifest["fitted_train_cell_ids"]) == set(split["train_cell_ids"])
    inputs = [json.loads(line) for line in (output / "test/inputs.jsonl").read_text().splitlines()]
    assert all(set(r) == {"example_id", "cell_id", "benchmark", "model_input"} for r in inputs)
    labels = [
        json.loads(line) for line in (output / "private/test_labels.jsonl").read_text().splitlines()
    ]
    assert {r["example_id"] for r in inputs} == {r["example_id"] for r in labels}
    raw_index = json.loads((output / "train/raw_trajectory_index.json").read_text())
    assert {r["cell_id"] for r in raw_index} == set(split["train_cell_ids"])
    for entry in raw_index:
        assert (
            entry["sha256"]
            == hashlib.sha256(
                (raw_root / "cells" / entry["cell_id"] / "solve_out_sanitized.txt").read_bytes()
            ).hexdigest()
        )
    with pytest.raises(FileExistsError):
        wm_prepare.prepare(sources=[source], output_dir=output, test_count=1, seed=123)


def test_previous_gsm_exclusions_preserved_without_touching_aime(tmp_path):
    previous = tmp_path / "old.jsonl"
    previous.write_text(json.dumps({"example_id": "r0-01/exp-01", "eligible": False}) + "\n")
    rows = [
        {"example_id": "r0-01/exp-01", "benchmark": "gsm8k", "audit": {"eligible": True}},
        {"example_id": "aime-r0-01/exp-01", "benchmark": "aime2025", "audit": {"eligible": True}},
    ]
    result = wm_prepare.apply_previous_gsm_audit(rows, previous)
    assert not rows[0]["audit"]["eligible"]
    assert rows[1]["audit"]["eligible"]
    assert result["excluded_ids"] == ["r0-01/exp-01"]


def test_eligibility_has_one_canonical_value_and_initial_screen_is_retained():
    rows = [{"eligible": True, "audit": {"eligible": False}}]
    wm_prepare.normalize_eligibility(rows)
    assert rows == [{"eligible": False, "initial_eligible": True, "audit": {"eligible": False}}]
    wm_prepare.normalize_eligibility(rows)
    assert rows[0]["initial_eligible"] is True


def history_rows():
    return [
        {
            "example_id": "a/exp-01",
            "cell_id": "a",
            "card_id": "exp-01",
            "model_input": {},
            "label": {"accuracy": 0.0, "evaluation_n": 30},
            "first_submitted_at": "2026-01-01T00:00:00Z",
        },
        {
            "example_id": "a/exp-02",
            "cell_id": "a",
            "card_id": "exp-02",
            "model_input": {},
            "label": {"accuracy": 0.5},
            "first_submitted_at": "2026-01-01T02:00:00Z",
            "prior_observations": [
                {
                    "example_id": "a/exp-01",
                    "stage": "closed",
                    "at": "2026-01-01T01:00:00Z",
                    "record": {"card": {"result": {"execution": "completed"}}},
                }
            ],
        },
    ]


def test_known_official_context_uses_prior_closed_nodes_and_keeps_zero():
    rows = history_rows()
    wm_prepare.add_known_checkpoint_history(rows)
    assert rows[0]["model_input"]["known_previous_checkpoints"] == []
    history = rows[1]["model_input"]["known_previous_checkpoints"]
    assert len(history) == 1 and history[0]["official_accuracy"] == 0.0
    assert history[0]["checkpoint_ref"] == "exp-01"


def test_future_closed_record_cannot_enter_known_context():
    rows = history_rows()
    rows[1]["prior_observations"][0]["at"] = "2026-01-01T03:00:00Z"
    with pytest.raises(ValueError, match="Future"):
        wm_prepare.add_known_checkpoint_history(rows)


def test_explicit_fidelity_exclusion_is_applied(tmp_path):
    rows = [{"example_id": "x", "audit": {"eligible": True}}]
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps({"decisions": {"x": {"action": "exclude", "reason": "changed epochs"}}})
    )
    wm_prepare.apply_fidelity_decisions(rows, path)
    assert not rows[0]["audit"]["eligible"]
    assert rows[0]["audit"]["setup_change_review"]["reason"] == "changed epochs"
