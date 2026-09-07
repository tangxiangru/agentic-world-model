"""Scoring checks for matched RPM comparisons using synthetic outcomes only."""

import json
import math
import sys

import pytest

from tools.outcome_prediction import rpm_compare as compare


def pair(cell_id, *, p=0.8, gap=0.2, positive=True):
    return {
        "cell_id": cell_id,
        "y_a": 0.5 + gap if positive else 0.5,
        "y_b": 0.5 if positive else 0.5 + gap,
        "gap": gap,
        "probabilities": {"method": p},
        "choices": {},
    }


@pytest.mark.parametrize("positive,expected_credit", [(True, 1.0), (False, 0.0)])
def test_frozen_probability_tie_uses_explicit_choice_while_learned_tie_gets_half_credit(
    positive, expected_credit
):
    row = pair("cell", p=0.5, gap=0.2, positive=positive)
    row["probabilities"] = {"frozen": 0.5, "learned": 0.5}
    row["choices"] = {"frozen": True}
    frozen = compare.per_cell([row], "frozen")["cell"]["mean"]
    learned = compare.per_cell([row], "learned")["cell"]["mean"]
    assert frozen[0] == expected_credit
    assert frozen[1] == pytest.approx((1 - expected_credit) * 0.2)
    assert learned[:2] == pytest.approx([0.5, 0.1])
    assert frozen[2:] == pytest.approx([0.25, math.log(2)])
    assert learned[2:] == pytest.approx(frozen[2:])


def test_explicit_false_choice_is_not_mistaken_for_missing_choice():
    row = pair("cell", p=0.5, positive=False)
    row["choices"] = {"method": False}
    assert compare.per_cell([row], "method")["cell"]["mean"][0] == 1.0


def test_cell_and_pair_weighting_give_expected_accuracy_and_regret():
    rows = [pair("many", p=0.8, gap=0.2) for _ in range(3)]
    rows.append(pair("one", p=0.2, gap=0.4))
    result = compare.summarize(rows, "method", bootstraps=200)
    assert result["pairs"] == 4
    assert result["cells"] == 2
    assert result["per_cell"]["many"]["n"] == 3
    assert result["per_cell"]["one"]["n"] == 1
    assert result["macro_accuracy"] == 0.5
    assert result["micro_accuracy"] == 0.75
    assert result["macro_regret"] == pytest.approx(0.2)
    assert result["micro_regret"] == pytest.approx(0.1)
    assert result["macro_brier"] == pytest.approx(0.34)
    assert result["micro_brier"] == pytest.approx(0.19)
    for prefix in ("macro", "micro"):
        for metric in ("accuracy", "regret", "brier", "log_loss"):
            estimate = result[f"{prefix}_{metric}"]
            lower, upper = result[f"{prefix}_{metric}_ci95"]
            assert lower <= estimate <= upper


def test_paired_bootstrap_positive_always_means_better_and_reverses_exactly():
    rows = [pair("first", gap=0.1), pair("second", gap=0.2)]
    for row in rows:
        row["probabilities"] = {"good": 0.8, "bad": 0.2}
    forward = compare.paired_difference(rows, "good", "bad", bootstraps=200)
    reverse = compare.paired_difference(rows, "bad", "good", bootstraps=200)
    expected = {
        "accuracy_gain": 1.0,
        "regret_reduction": 0.15,
        "brier_reduction": 0.60,
        "log_loss_reduction": math.log(4),
    }
    assert forward["pairs"] == 2 and forward["cells"] == 2
    for prefix in ("macro", "micro"):
        for metric, value in expected.items():
            key = f"{prefix}_{metric}"
            assert forward[key] == pytest.approx(value)
            assert reverse[key] == pytest.approx(-value)
            lower, upper = forward[key + "_ci95"]
            assert lower > 0
            assert reverse[key + "_ci95"] == pytest.approx([-upper, -lower])


def test_paired_bootstrap_resamples_cells_and_retains_pair_weights():
    rows = [pair("many") for _ in range(3)] + [pair("one")]
    for row in rows:
        row["probabilities"] = {
            "method": 0.8 if row["cell_id"] == "many" else 0.2,
            "reference": 0.2 if row["cell_id"] == "many" else 0.8,
        }
    result = compare.paired_difference(rows, "method", "reference", bootstraps=200)
    assert result["macro_accuracy_gain"] == pytest.approx(0)
    assert result["micro_accuracy_gain"] == pytest.approx(0.5)
    assert result["macro_regret_reduction"] == pytest.approx(0)
    assert result["micro_regret_reduction"] == pytest.approx(0.1)


def write_fixture(root, *, learned_fold=0, failed_second=False):
    rows, labels, learned = [], [], []
    for index in range(2):
        cell = f"cell-{index}"
        a_id, b_id = f"{cell}/exp-01", f"{cell}/exp-02"
        rows.extend(
            [
                {
                    "example_id": a_id,
                    "first_submitted_at": "2026-09-04T01:00:00Z",
                    "scientist_model": "fixture-scientist",
                },
                {
                    "example_id": b_id,
                    "first_submitted_at": "2026-09-04T02:00:00Z",
                    "scientist_model": "fixture-scientist",
                },
            ]
        )
        labels.append(
            {
                "id": f"pair-{index}",
                "a_id": a_id,
                "b_id": b_id,
                "cell_id": cell,
                "y_a": 0.8,
                "y_b": 0.6,
                "gap": 0.2,
                "fold": 0,
                "history_count": index,
                "swapped": bool(index),
            }
        )
        learned.append(
            {
                "a_id": a_id,
                "b_id": b_id,
                "fold": learned_fold,
                "probabilities": {"fixed_recipe_numeric_logistic_C1": 0.8, "random_choice": 0.0},
            }
        )
        for arm in ("within_run", "cross_run"):
            input_path = root / f"judge/inputs/{arm}/pair-{index}.json"
            payload = {"task": "synthetic fixture", "index": index, "arm": arm}
            compare.write_json(input_path, payload)
            invalid = failed_second and index == 1 and arm == "cross_run"
            output = {
                "id": f"pair-{index}",
                "input": str(input_path),
                "input_sha256": compare.digest(payload),
                "valid": not invalid,
                "elapsed_seconds": 2.0,
                "model_requested": "fixture-judge",
                "p_a": 0.8,
                "choice_a": True,
            }
            if invalid:
                output["error"] = "synthetic failure"
            else:
                output["cost_usd"] = 0.1
            compare.write_json(root / f"judge/outputs/{arm}/pair-{index}.json", output)
    examples = root / "examples.jsonl"
    examples.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    compare.write_json(root / "judge/hidden_labels.json", labels)
    (root / "learned").mkdir()
    (root / "learned/pair_predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in learned) + "\n"
    )
    return examples


def test_learned_and_judge_fold_mismatch_invalidates_comparison(tmp_path):
    examples = write_fixture(tmp_path, learned_fold=1)
    with pytest.raises(ValueError, match="folds disagree"):
        compare.combine(tmp_path, examples)


def test_changed_frozen_input_invalidates_comparison(tmp_path):
    examples = write_fixture(tmp_path)
    compare.write_json(tmp_path / "judge/inputs/within_run/pair-0.json", {"changed": True})
    with pytest.raises(ValueError, match="input hash"):
        compare.combine(tmp_path, examples)


def test_one_failed_arm_excludes_pair_from_every_matched_method_and_counts_failure(
    tmp_path, monkeypatch
):
    examples = write_fixture(tmp_path, failed_second=True)
    records, diagnostics = compare.combine(tmp_path, examples)
    assert len(records) == 2
    assert records[0]["valid_arms"] == ["within_run", "cross_run"]
    assert records[1]["valid_arms"] == ["within_run"]
    assert "frozen/cross_run" not in records[1]["probabilities"]
    assert "fixed_half_blend" not in records[1]["probabilities"]
    assert "random_choice" not in records[0]["probabilities"]
    assert records[0]["probabilities"]["later_recipe"] == 0.0
    assert records[0]["probabilities"]["fixed_half_blend"] == 0.8
    assert diagnostics["arms"]["within_run"]["attempted"] == 2
    cross = diagnostics["arms"]["cross_run"]
    assert cross["attempted"] == 2 and cross["valid"] == 1
    assert cross["reported_cost_usd"] == pytest.approx(0.1)
    assert cross["unknown_cost_calls"] == 1
    assert cross["elapsed_call_seconds"] == 4.0
    assert cross["invalid"] == [{"id": "pair-1", "error": "synthetic failure"}]
    monkeypatch.setattr(
        sys, "argv", ["rpm_compare", "--root", str(tmp_path), "--examples", str(examples)]
    )
    compare.main()
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["diagnostics"]["total_pairs"] == 2
    assert report["diagnostics"]["matched_pairs"] == 1
    assert report["diagnostics"]["matched_cells"] == 1
    assert report["diagnostics"]["displayed_order"] == {"swapped": 1, "not_swapped": 1}
    for metrics in report["metrics"]["matched_gap_01"].values():
        assert metrics["pairs"] == 1 and metrics["cells"] == 1
    assert len((tmp_path / "comparison_pairs.jsonl").read_text().splitlines()) == 2
