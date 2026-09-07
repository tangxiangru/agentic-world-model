"""Guard the information boundary used by the exploratory prediction study."""

import copy
import json

import pytest

from tools.outcome_prediction.build_examples import build, canonical, unique_largest_measurement


def recipe():
    return {
        "setup": {
            "base_model": "google/gemma-3-4b-pt",
            "parent_checkpoint": {"origin": "base_model", "path": "google/gemma-3-4b-pt"},
            "method": {
                "family": "sft",
                "framework": "transformers 4.57.3 Trainer from /home/ben/pylibs train_exp01.py",
                "hyperparams": {
                    "lr": 2e-5,
                    "epochs": 1,
                    "precision": "bf16",
                    "other": "Already scored 0.91",
                },
            },
            "data": [
                {
                    "source": "openai/gsm8k and data/sft_v1.jsonl from ckpts/exp-01",
                    "n_examples": 7473,
                }
            ],
            "command": {
                "argv": ["python", "train_exp01.py", "--lr", "2e-5"],
                "script": "/home/ben/task/train_exp01.py",
            },
            "output_dir": "ckpts/exp01",
            "budget": {"planned_h": 2},
        },
        "problem": {"statement": "Accuracy 0.92"},
        "hypothesis": {"claim": "should score 0.99"},
        "result": {"measurements": []},
    }


def test_canonical_excludes_scores_identity_and_local_paths():
    card = recipe()
    features = canonical(card)
    encoded = json.dumps(features)
    for forbidden in ["0.91", "0.92", "0.99", "exp01", "exp-01", "pylibs", "sft_v1", "train_exp"]:
        assert forbidden not in encoded
    assert "openai/gsm8k" in encoded
    changed = copy.deepcopy(card)
    changed["result"] = {
        "execution": "failed",
        "measurements": [{"name": "accuracy", "value": 0.04}],
    }
    changed["conclusion"] = {"verdict": "contradicted"}
    changed["problem"] = {"statement": "Different observed outcome"}
    assert canonical(changed) == features


def test_merge_intervention_is_retained():
    card = recipe()
    card["setup"]["method"]["family"] = "merge"
    card["setup"]["command"]["argv"] = [
        "python",
        "soup.py",
        "--models",
        "ckpts/a/checkpoint-100",
        "ckpts/b/final",
        "--weights",
        "1.5",
        "-0.5",
    ]
    obj, numeric, _ = canonical(card)
    assert obj["merge"] == {"member_count": 2, "weights": [1.5, -0.5], "intermediate_steps": [100]}
    assert numeric["merge_weight_1"] == -0.5
    assert "ckpts" not in json.dumps(obj)


def test_ambiguous_prior_scores_are_not_optimistically_selected():
    card = {
        "result": {
            "measurements": [
                {"name": "accuracy_greedy", "n": 100, "value": 0.5},
                {"name": "accuracy_sample", "n": 100, "value": 0.8},
            ]
        }
    }
    assert unique_largest_measurement(card) is None
    card["result"]["measurements"].append({"name": "accuracy", "n": 200, "value": 0.6})
    assert unique_largest_measurement(card) == 0.6


def stage_fixture(tmp_path, first_stage="plan", changed=False):
    (tmp_path / "manifest_r0.json").write_text(
        json.dumps([{"cell_id": "r0-99", "scientist_model": "fixture"}])
    )
    cell = tmp_path / "cells/r0-99"
    cards = cell / "wm/cards/exp-01"
    cards.mkdir(parents=True)
    card = recipe()
    first = {"at": "2026-09-04T00:00:00Z", "event": "submit", "card": card}
    (cards / "record-01.json").write_text(json.dumps(first))
    final = copy.deepcopy(card)
    final["result"] = {
        "execution": "completed",
        "output_checkpoint": "ckpts/exp01",
        "measurements": [{"name": "accuracy", "n": 1319, "value": 0.4}],
    }
    if changed:
        final["setup"]["method"]["hyperparams"]["epochs"] = 2
    (cards / "card.json").write_text(json.dumps(final))
    (cell / "wm/records.jsonl").write_text(
        json.dumps({"event": "submit", "stage": first_stage, "card_id": "exp-01", "missing": []})
        + "\n"
    )
    (cell / "wm_metrics").mkdir()
    (cell / "wm_metrics/exp-01.json").write_text(json.dumps({"accuracy": 0.4}))
    return cell


@pytest.mark.parametrize(
    "stage,changed,reason",
    [
        ("closed", False, "first_submission_not_plan"),
        ("plan", True, "canonical_recipe_changed_after_first_submission"),
    ],
)
def test_retrospective_or_changed_recipe_is_excluded(tmp_path, stage, changed, reason):
    stage_fixture(tmp_path, stage, changed)
    rows, _ = build(tmp_path)
    assert not rows[0]["eligible"]
    assert reason in rows[0]["exclusion_reasons"]


def test_changing_official_label_never_changes_recipe_features(tmp_path):
    cell = stage_fixture(tmp_path)
    before = build(tmp_path)[0][0]
    (cell / "wm_metrics/exp-01.json").write_text(json.dumps({"accuracy": 0.95}))
    after = build(tmp_path)[0][0]
    assert before["y"] != after["y"]
    for key in [
        "recipe_text",
        "lineage_text",
        "numeric_features",
        "categorical_features",
        "eligible",
    ]:
        assert before[key] == after[key]


def test_grouped_folds_never_split_a_run():
    pytest.importorskip("sklearn")
    from tools.outcome_prediction.benchmark import split_groups

    rows = [{"cell_id": f"cell-{i // 3}", "y": i / 30} for i in range(30)]
    seen = []
    for train, test in split_groups(rows, 5, 42):
        assert {rows[i]["cell_id"] for i in train}.isdisjoint({rows[i]["cell_id"] for i in test})
        seen.extend(test)
    assert sorted(seen) == list(range(len(rows)))
