import copy
import json
from types import SimpleNamespace

import pytest

from tools.outcome_prediction import rpm_faithful as faithful
from tools.outcome_prediction.rpm_judge import digest, write_json


def test_swapping_updates_weight_and_data_membership_and_is_involution():
    payload = {
        "scored_predecessors": [
            {
                "weight_ancestor_of": ["candidate_A", "candidate_B"],
                "weight_or_data_ancestor_of": ["candidate_B"],
            }
        ]
    }
    original = copy.deepcopy(payload)
    faithful.swap_candidate_context(payload)
    assert payload["scored_predecessors"][0]["weight_or_data_ancestor_of"] == ["candidate_A"]
    faithful.swap_candidate_context(payload)
    assert payload == original


def test_anonymization_removes_audit_identifiers_without_scrubbing_numbers():
    result = faithful.anonymize_packet(
        {
            "earliest_plan_source": "/raw/r0-01/exp-01/record-01.json",
            "plan": "exp-01 uses lr 0.0001, warmup 0.03; train.py",
            "ref": "exp_01",
            "nested": {"local_record_at": "now", "accuracy": 0.73},
        },
        "r0-01",
        ["exp-01"],
    )
    assert "earliest_plan_source" not in result
    assert result["nested"] == {"accuracy": 0.73}
    assert "0.0001" in result["plan"] and "0.03" in result["plan"]
    assert result["ref"] in result["plan"]
    assert "exp-01" not in str(result)


def test_redecode_preserves_raw_output_and_makes_no_model_call(tmp_path, monkeypatch):
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Decoding must not make a model call")

    monkeypatch.setattr(faithful, "invoke", forbidden_call)
    original = {
        "id": "test-pair",
        "swapped": True,
        "valid": False,
        "raw_response": {
            "subtype": "success",
            "result": r"The code handles `\boxed{}`. Final: \boxed{B}",
        },
    }
    path = tmp_path / "outputs" / "test-pair.json"
    write_json(path, original)
    original_bytes = path.read_bytes()
    faithful.redecode(tmp_path)
    assert path.read_bytes() == original_bytes
    result = faithful.read_verdict(tmp_path, "test-pair")
    assert result["valid"] and result["choice_a"]
    assert not result["original_decode_valid"]
    assert result["raw_response"] == original["raw_response"]


def test_derived_verdict_cannot_replace_raw_response(tmp_path):
    write_json(tmp_path / "outputs/p.json", {"raw_response": {"result": "original"}})
    write_json(tmp_path / "decoded_outputs/p.json", {"raw_response": {"result": "changed"}})
    with pytest.raises(ValueError, match="original raw response"):
        faithful.read_verdict(tmp_path, "p")


def test_frozen_model_mismatch_fails_before_calls(tmp_path):
    write_json(
        tmp_path / "protocol.json",
        {"isolation_sha256": digest(faithful.ISOLATION), "model": "expected"},
    )
    with pytest.raises(ValueError, match="model differs"):
        faithful.execute(SimpleNamespace(output_dir=tmp_path, model="wrong"))


def test_fixed_models_hold_out_whole_runs_and_use_no_label_payload(tmp_path):
    labels = []
    for i in range(4):
        identity = f"pair-{i}"
        labels.append(
            {
                "id": identity,
                "fold": i % 2,
                "cell_id": f"run-{i}",
                "swapped": bool(i % 2),
                "y_a": 0.4 + 0.05 * i,
                "y_b": 0.5,
            }
        )
        write_json(
            tmp_path / "inputs" / (identity + ".json"),
            {
                "candidate_A": {"plan": "small data low learning rate", "code": "train(100)"},
                "candidate_B": {"plan": "large data high learning rate", "code": "train(200)"},
                "history": [{"official_accuracy": 0.3}],
            },
        )
    write_json(tmp_path / "hidden_labels.json", labels)
    faithful.fit_rich_rankers(tmp_path)
    predictions = json.loads((tmp_path / "rich_learned.json").read_text())
    folds = json.loads((tmp_path / "rich_learned_folds.json").read_text())
    assert len(predictions) == 4
    for fold in folds:
        train = {p["cell_id"] for p in labels if p["id"] in fold["train_pairs"]}
        test = {p["cell_id"] for p in labels if p["id"] in fold["test_pairs"]}
        assert not train.intersection(test)
    assert all(0 <= p <= 1 for row in predictions for p in row["probabilities"].values())
