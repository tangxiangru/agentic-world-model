"""Synthetic checks only; no real-data fitting or network."""

import copy

import numpy as np
import pytest

from tools.outcome_prediction import wm_one_step_train as train


def examples(n=15):
    rows, labels = [], {}
    for i in range(n):
        key, reference, target = f"fixture-{i // 3}/exp-{i}", 0.2, 0.1 + i / 30
        rows.append({
            "example_id": key, "cell_id": f"fixture-{i // 3}",
            "benchmark": "fixture", "scientist_model": "fixture",
            "split": "train", "parent_reference": reference,
            "reference_kind": "measured_parent",
            "views": {v: {"parent_accuracy": reference, "setting": i / n} for v in ("parent", "current", "history")},
        })
        labels[key] = {"accuracy": target, "delta_accuracy": target - reference}
    return rows, labels


def heldout():
    rows, _ = examples(1)
    rows[0].update(example_id="heldout/exp-01", cell_id="heldout", split="test")
    return rows


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1, True, "0.2"])
@pytest.mark.parametrize("owner", ["target", "parent"])
def test_invalid_accuracy_never_reaches_fit(value, owner):
    rows, labels = examples()
    if owner == "target":
        labels[rows[0]["example_id"]]["accuracy"] = value
    else:
        rows[0]["parent_reference"] = value
    with pytest.raises(ValueError, match="accuracies"):
        train.select_and_fit(rows, labels, heldout())


def test_missing_delta_rejected_even_with_two_accuracies():
    rows, labels = examples()
    del labels[rows[0]["example_id"]]["delta_accuracy"]
    with pytest.raises(ValueError, match="delta label"):
        train.validate_rows(rows, labels)


def test_zero_accuracy_is_valid():
    rows, labels = examples(1)
    rows[0]["parent_reference"] = 0
    labels[rows[0]["example_id"]] = {"accuracy": 0.0, "delta_accuracy": 0.0}
    train.validate_rows(rows, labels)


def test_session_overlap_and_extra_labels_rejected():
    rows, labels = examples()
    test = heldout()
    test[0]["cell_id"] = rows[0]["cell_id"]
    with pytest.raises(ValueError, match="session overlap"):
        train.select_and_fit(rows, labels, test)
    labels["heldout/exp-01"] = {"accuracy": 0.3, "delta_accuracy": 0.1}
    with pytest.raises(ValueError, match="identities"):
        train.select_and_fit(rows, labels, heldout())


def test_constant_delta_prediction_is_not_missing_reference_imputation():
    rows, labels = examples(2)
    for row in rows:
        labels[row["example_id"]] = {"accuracy": 0.3, "delta_accuracy": 0.3 - 0.2}
    result = train.baseline_predictions(rows, labels, heldout())
    assert result["parent_unchanged"].tolist() == [0.2]
    assert result["mean_delta"].tolist() == pytest.approx([0.3])
    test = heldout()
    test[0]["parent_reference"] = None
    with pytest.raises(ValueError):
        train.baseline_predictions(rows, labels, test)


def test_grouped_selection_only_uses_training_labels(monkeypatch):
    rows, labels = examples()
    snapshots = copy.deepcopy((rows, labels))
    calls = []

    def fake_suite(fit, fit_labels, valid, model_dir=None):
        assert set(fit_labels) == {r["example_id"] for r in fit}
        assert not ({r["cell_id"] for r in fit} & {r["cell_id"] for r in valid})
        assert "heldout/exp-01" not in fit_labels
        calls.append((len(fit), len(valid)))
        return {name: np.asarray([r["parent_reference"] for r in valid]) for name in train.ARMS}, {}

    monkeypatch.setattr(train, "fit_suite", fake_suite)
    forecasts, details = train.select_and_fit(rows, labels, heldout())
    assert len(calls) == 6  # five grouped inner folds and a final TRAIN fit
    assert len(details["oof_predictions"]) == len(rows)
    assert details["selected"] == min(train.SELECTABLE)
    np.testing.assert_array_equal(forecasts[train.PRIMARY], forecasts[details["selected"]])
    assert (rows, labels) == snapshots
    assert all(not name.startswith("absolute_") for name in train.SELECTABLE)


def test_all_fixed_model_families_fit_synthetic_complete_labels(tmp_path):
    rows, labels = examples()
    predictions, records = train.fit_suite(rows, labels, heldout(), tmp_path)
    assert set(predictions) == set(train.ARMS)
    assert set(records) == set(train.SPECS)
    for value in predictions.values():
        assert value.shape == (1,)
        assert np.isfinite(value).all()
        assert ((value >= 0) & (value <= 1)).all()
    assert len(list(tmp_path.glob("*.joblib"))) == len(train.SPECS)


def test_measured_sensitivity_never_includes_base_examples():
    rows, _ = examples()
    rows[-1]["split"] = "test"
    # Keep the entire held-out session out of train.
    for row in rows:
        if row["cell_id"] == rows[-1]["cell_id"]:
            row["split"] = "test"
    rows[0]["reference_kind"] = "published_base"
    regimes = train.regimes_for(rows)
    assert rows[0]["example_id"] in regimes["combined__fixture"]["train_ids"]
    assert rows[0]["example_id"] not in regimes["measured_parent__fixture"]["train_ids"]
