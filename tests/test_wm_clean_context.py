"""Synthetic exact-checkpoint context tests; no model fits or network calls."""

import copy
import hashlib
import json

import pytest

from tools.outcome_prediction.wm_clean_context import _project_line, build_context
from tools.outcome_prediction.wm_clean_refresh import freeze_dataset


def row(number, *, cell="aime2-r0-90", parent="org/base", output=None, label=0.25):
    card = f"exp-{number:02d}"
    out = output or f"/task/ckpts/{card}"
    return {
        "example_id": cell + "/" + card,
        "cell_id": cell,
        "card_id": card,
        "benchmark": "aime2025",
        "scientist_model": "fixture",
        "first_stage": "plan",
        "first_submitted_at": f"2026-01-01T{number:02d}:00:00Z",
        "eligible": True,
        "model_input": {
            "task": {"base_model": "org/base"},
            "plan": {
                "setup": {
                    "base_model": "org/base",
                    "output_dir": out,
                    "parent_checkpoint": {"path": parent, "origin": "base_model"},
                    "command": {
                        "argv": ["python", "train.py", "--model", parent],
                        "cwd": "/task",
                        "script": "/task/train.py",
                    },
                    "method": {
                        "family": "sft",
                        "hyperparams": {"lr": 1e-5 * number, "epochs": number},
                    },
                    "data": [],
                }
            },
            "code": [],
        },
        "prior_observations": [],
        "label": {"accuracy": label, "official_metric": {"accuracy": label}, "evaluation_n": 30},
        "audit": {
            "eligible": True,
            "reasons": [],
            "first_missing_fields": [],
            "label_fidelity_flags": [],
            "first_final_setup_changed": False,
            "first_final_setup_changed_fields": [],
            "output_artifact_path_mismatch": False,
            "output_artifact_comparison": {
                "first_declared_output_dir": out,
                "final_output_checkpoint": out,
            },
        },
        "provenance": {
            "manifest_name": "fixture.json",
            "first_record_sha256": "1" * 64,
            "card_sha256": "2" * 64,
            "official_metric_sha256": "3" * 64,
        },
    }


def close(parent, target, *, at=None, output=None):
    at = at or parent["first_submitted_at"].replace(":00:00Z", ":30:00Z")
    target["prior_observations"].append(
        {
            "example_id": parent["example_id"],
            "stage": "closed",
            "at": at,
            "measurements": [{"value": 0.999, "metric": "DO_NOT_DECODE"}],
            "record": {
                "at": at,
                "card": {
                    "card_id": parent["card_id"],
                    "result": {
                        "output_checkpoint": output
                        or parent["model_input"]["plan"]["setup"]["output_dir"],
                        "measurements": [{"value": 0.888, "metric": "DO_NOT_DECODE"}],
                    },
                },
            },
            "provenance": {"sha256": "4" * 64},
        }
    )


def build(tmp_path, rows, name="bundle"):
    bundle = tmp_path / name
    freeze_dataset(rows, bundle, test_count=int(len({r["cell_id"] for r in rows}) > 1))
    return build_context(bundle)


def by_id(examples):
    return {r["example_id"]: r for r in examples}


def test_exact_parent_and_full_recipe_history(tmp_path):
    a = row(1, label=0.2)
    b = row(2, parent="ckpts/exp-01", label=0.4)
    c = row(3, parent="ckpts/exp-02", label=0.6)
    close(a, b)
    close(a, c)
    close(b, c)
    examples, labels, audit = build(tmp_path, [a, b, c])
    rows = by_id(examples)
    last = rows[c["example_id"]]
    assert len(examples) == len(labels) == 3
    assert last["parent_reference"] == 0.4
    assert last["reference_known"] is True
    assert last["parent_ids"] == [b["example_id"]]
    assert last["history_ids"] == [a["example_id"], b["example_id"]]
    assert last["views"]["history"]["history.count"] == 2
    assert last["views"]["parent"]["action_minus_latest.hp.epochs"] == 1
    assert last["audit"]["history_scores_used"] is False
    assert audit["targets"] == 3


def test_base_and_unknown_are_zero_centering_not_observed_accuracy(tmp_path):
    a = row(1)
    b = row(2, parent="/unknown/weights")
    examples, _, _ = build(tmp_path, [a, b])
    rows = by_id(examples)
    for r in rows.values():
        assert r["parent_reference"] == 0
        assert r["reference_known"] is False
        assert r["views"]["current"]["reference_known"] == 0
    assert rows[a["example_id"]]["from_base"] is True
    assert rows[b["example_id"]]["from_base"] is False


def test_intermediate_checkpoint_never_uses_producing_card_final_score(tmp_path):
    a = row(1, label=0.7)
    b = row(2, parent="ckpts/exp-01/checkpoint-100")
    close(a, b)
    examples, _, _ = build(tmp_path, [a, b])
    last = by_id(examples)[b["example_id"]]
    assert not last["reference_known"]
    assert last["parent_ids"] == last["history_ids"] == []


def test_command_disagreement_never_uses_origin_or_parent_path(tmp_path):
    a = row(1, label=0.7)
    b = row(2, parent="/wrong/checkpoint")
    b["model_input"]["plan"]["setup"]["parent_checkpoint"] = {
        "path": "/task/ckpts/exp-01",
        "origin": "exp-01",
    }
    close(a, b)
    examples, _, _ = build(tmp_path, [a, b])
    assert not by_id(examples)[b["example_id"]]["reference_known"]


def test_explicit_parent_path_allowed_when_no_command_weight_override(tmp_path):
    a = row(1)
    b = row(2, parent="ckpts/exp-01")
    b["model_input"]["plan"]["setup"]["command"]["argv"] = ["python", "train.py"]
    close(a, b)
    examples, _, _ = build(tmp_path, [a, b])
    assert by_id(examples)[b["example_id"]]["reference_known"]


@pytest.mark.parametrize("at", ["2026-01-01T02:00:00Z", "2026-01-01T02:01:00Z"])
def test_close_at_or_after_child_is_not_available(tmp_path, at):
    a = row(1)
    b = row(2, parent="ckpts/exp-01")
    close(a, b, at=at)
    examples, _, _ = build(tmp_path, [a, b])
    assert not by_id(examples)[b["example_id"]]["reference_known"]


def test_mismatched_closed_output_cannot_supply_score(tmp_path):
    a = row(1)
    b = row(2, parent="ckpts/exp-01")
    close(a, b, output="/task/different")
    examples, _, _ = build(tmp_path, [a, b])
    assert not by_id(examples)[b["example_id"]]["reference_known"]


def test_missing_close_is_unknown_not_dropped(tmp_path):
    a = row(1)
    b = row(2, parent="ckpts/exp-01")
    examples, _, _ = build(tmp_path, [a, b])
    assert len(examples) == 2
    assert not by_id(examples)[b["example_id"]]["reference_known"]


@pytest.mark.parametrize("corruption", ["dirty", "unlabeled"])
def test_quarantined_or_unlabeled_parent_supplies_neither_score_nor_recipe(tmp_path, corruption):
    a = row(1, label=0.9)
    b = row(2, parent="ckpts/exp-01")
    close(a, b)
    if corruption == "dirty":
        a["audit"]["first_final_setup_changed"] = True
    else:
        a["label"] = {"accuracy": None, "official_metric": None}
    examples, labels, _ = build(tmp_path, [a, b])
    assert len(examples) == len(labels) == 1
    assert examples[0]["parent_ids"] == examples[0]["history_ids"] == []
    assert not examples[0]["reference_known"]


def test_duplicate_output_owner_is_ambiguous_even_if_one_is_dirty(tmp_path):
    a = row(1, output="/task/shared")
    b = row(2, output="/task/shared")
    b["audit"]["first_final_setup_changed"] = True
    c = row(3, parent="/task/shared")
    close(a, c)
    close(b, c)
    examples, _, _ = build(tmp_path, [a, b, c])
    assert not by_id(examples)[c["example_id"]]["reference_known"]


def test_other_session_cannot_supply_parent_score(tmp_path):
    a = row(1, cell="aime2-r0-91")
    b = row(2, cell="aime2-r0-92", parent="ckpts/exp-01")
    close(a, b)
    examples, _, _ = build(tmp_path, [a, b])
    assert not by_id(examples)[b["example_id"]]["reference_known"]


def test_multiple_merge_parents_are_context_but_not_average_score(tmp_path):
    a = row(1, label=0.2)
    b = row(2, label=0.8)
    c = row(3)
    setup = c["model_input"]["plan"]["setup"]
    setup["method"]["family"] = "merge"
    setup["command"]["argv"] = ["python", "train.py", "--models", "ckpts/exp-01", "ckpts/exp-02"]
    close(a, c)
    close(b, c)
    examples, _, _ = build(tmp_path, [a, b, c])
    last = by_id(examples)[c["example_id"]]
    assert last["parent_ids"] == [a["example_id"], b["example_id"]]
    assert last["parent_reference"] == 0 and not last["reference_known"]
    assert last["views"]["parent"]["history.count"] == 2


def test_grandparent_score_changes_do_not_change_grandchild_features(tmp_path):
    a = row(1, label=0.2)
    b = row(2, parent="ckpts/exp-01", label=0.4)
    c = row(3, parent="ckpts/exp-02")
    close(a, b)
    close(b, c)
    first, _, _ = build(tmp_path, [a, b, c], "one")
    altered = copy.deepcopy(a)
    altered["label"] = {"accuracy": 0.9, "official_metric": {"accuracy": 0.9}}
    second, _, _ = build(tmp_path, [altered, b, c], "two")
    assert by_id(first)[c["example_id"]]["views"] == by_id(second)[c["example_id"]]["views"]


def test_private_labels_and_prior_measurements_never_json_decoded(monkeypatch):
    a = row(1)
    b = row(2, parent="ckpts/exp-01")
    close(a, b)
    b["label"] = {"FORBIDDEN_LABEL_SENTINEL": 999}
    text = json.dumps(b)
    original = json.loads

    def checked(raw, *args, **kwargs):
        assert "FORBIDDEN_LABEL_SENTINEL" not in raw
        assert "DO_NOT_DECODE" not in raw
        return original(raw, *args, **kwargs)

    monkeypatch.setattr("tools.outcome_prediction.wm_clean_context.json.loads", checked)
    projected = _project_line(text, {a["example_id"], b["example_id"]})
    assert projected["earlier_closed"][0]["output_checkpoint"] == "/task/ckpts/exp-01"


def test_unavailable_command_keeps_target_and_unknown_base(tmp_path):
    a = row(1)
    a["model_input"]["plan"]["setup"]["command"]["argv"] = ["bash", "-c", "python train.py"]
    examples, _, _ = build(tmp_path, [a])
    assert len(examples) == 1
    assert examples[0]["from_base"] is None
    assert not examples[0]["reference_known"]


def test_official_zero_parent_is_known(tmp_path):
    a = row(1, label=0.0)
    b = row(2, parent="ckpts/exp-01")
    close(a, b)
    examples, _, _ = build(tmp_path, [a, b])
    last = by_id(examples)[b["example_id"]]
    assert last["reference_known"] is True and last["parent_reference"] == 0


def test_partition_and_current_target_label_separation(tmp_path):
    examples, labels, _ = build(tmp_path, [row(1, cell="aime2-r0-90"), row(1, cell="aime2-r0-91")])
    assert {r["partition"] for r in examples} == {"train", "test"}
    assert set(labels) == {r["example_id"] for r in examples}
    for r in examples:
        assert "label" not in r and "accuracy" not in r
        for f in r["views"].values():
            assert all(v is None or isinstance(v, (int, float)) for v in f.values())
            assert not any(r["cell_id"] in k for k in f)


def test_inventory_hash_must_be_in_manifest(tmp_path):
    bundle = tmp_path / "bundle"
    freeze_dataset([row(1)], bundle, test_count=0)
    manifest = json.loads((bundle / "manifest.json").read_text())
    del manifest["private/inventory.jsonl"]
    # Controlled synthetic artifact corruption; no real frozen files touched.
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Private inventory"):
        build_context(bundle)


def test_modified_inventory_rejected_by_loader(tmp_path):
    bundle = tmp_path / "bundle"
    freeze_dataset([row(1)], bundle, test_count=0)
    path = bundle / "private/inventory.jsonl"
    path.write_text(path.read_text() + "\n")
    assert hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Modified frozen"):
        build_context(bundle)
