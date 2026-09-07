"""Synthetic-only checks for agent-selected local numerical fitting."""

import copy
import json

import numpy as np
import pytest

from tools.outcome_prediction import wm_agent_local_fit as local


def fixture(n=12):
    rows, labels = [], {}
    for i in range(n):
        reference = 0.1 + i / 100
        features = {key: 0.0 for key in local.FEATURE_KEYS}
        features.update({
            "parent.accuracy": reference, "parent.reference.measured": 1.0,
            "current.config.epochs": float(i % 4),
            "current.config.learning_rate": 0.00002,
        })
        row = {
            "example_id": f"train-{i // 2}/exp-{i}", "cell_id": f"train-{i // 2}",
            "benchmark": "synthetic", "split": "train", "parent_reference": reference,
            "reference_kind": "measured_parent", "views": {"history": features},
        }
        rows.append(row)
        target = reference + 0.03 * (i % 4)
        labels[row["example_id"]] = {"accuracy": target, "delta_accuracy": target - reference}
    query = copy.deepcopy(rows[0])
    query.update(example_id="heldout/exp-0", cell_id="heldout", split="test")
    selection = {
        "selected_ids": [row["example_id"] for row in rows], "weights": [1.0] * n,
        "model": "ridge_10", "feature_keys": ["parent.accuracy", "current.config.epochs"],
        "rationale": "Synthetic test selection only.",
    }
    return rows, labels, query, selection


def test_exact_whitelist_matches_frozen_extractor():
    from tools.outcome_prediction.wm_one_step_features import features

    encoded = features({
        "parent": {"kind": "published_base", "accuracy": 0.0},
        "model_input": {}, "history": [], "history_complete_to_base": True,
    }, mode="history")
    assert len(local.FEATURE_KEYS) == 151
    assert set(local.FEATURE_KEYS) == set(encoded)


@pytest.mark.parametrize("model", local.MODELS)
def test_all_models_deterministic_and_json_reproducible(model):
    rows, labels, query, selection = fixture()
    selection["model"] = model
    selection["weights"] = [1 + i % 3 for i in range(len(rows))]
    if model == "weighted_median":
        selection["feature_keys"] = []
    before = copy.deepcopy((rows, labels, query, selection))
    result = local.fit_predict(rows, labels, query, selection)
    assert result == local.fit_predict(rows, labels, query, selection)
    saved = json.loads(json.dumps(result, allow_nan=False))
    replay = local.predict_from_state(saved["model_state"], query)
    assert all(result[key] == value for key, value in replay.items())
    assert 0 <= result["predicted_accuracy"] <= 1
    assert result["predicted_delta"] == result["predicted_accuracy"] - query["parent_reference"]
    assert result["fit_metadata"]["training_sessions"] == 6
    assert result["conditional_training_diagnostics"]["equal_session_mae"] >= 0
    assert before == (rows, labels, query, selection)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1, True, "0.2"])
@pytest.mark.parametrize("owner", ["target", "parent", "query"])
def test_invalid_or_missing_accuracy_is_rejected(value, owner):
    rows, labels, query, selection = fixture()
    if owner == "target":
        labels[rows[0]["example_id"]]["accuracy"] = value
    elif owner == "parent":
        rows[0]["parent_reference"] = value
    else:
        query["parent_reference"] = value
    with pytest.raises(ValueError, match="Accuracy"):
        local.fit_predict(rows, labels, query, selection)


def test_genuine_zero_is_valid_and_missing_delta_is_not():
    rows, labels, query, selection = fixture()
    rows[0]["parent_reference"] = 0.0
    rows[0]["views"]["history"]["parent.accuracy"] = 0.0
    labels[rows[0]["example_id"]] = {"accuracy": 0.0, "delta_accuracy": 0.0}
    local.fit_predict(rows, labels, query, selection)
    del labels[rows[0]["example_id"]]["delta_accuracy"]
    with pytest.raises(ValueError, match="Delta label"):
        local.fit_predict(rows, labels, query, selection)


@pytest.mark.parametrize("change", [
    "extra_label", "query_label", "row_label", "heldout_row", "session_overlap",
    "benchmark", "unknown_reference", "mismatched_parent_feature", "feature_injection",
    "feature_nan", "missing_feature", "wrong_delta", "duplicate_available",
])
def test_input_leakage_and_invalid_labels_fail_closed(change):
    rows, labels, query, selection = fixture()
    if change == "extra_label":
        labels[query["example_id"]] = {"accuracy": 0.99, "delta_accuracy": 0.89}
    elif change == "query_label":
        query["accuracy"] = 0.99
    elif change == "row_label":
        rows[0]["accuracy"] = 0.99
    elif change == "heldout_row":
        rows[0]["split"] = "test"
    elif change == "session_overlap":
        query["cell_id"] = rows[0]["cell_id"]
    elif change == "benchmark":
        query["benchmark"] = "different"
    elif change == "unknown_reference":
        rows[0]["reference_kind"] = "unknown"
    elif change == "mismatched_parent_feature":
        rows[0]["views"]["history"]["parent.accuracy"] = 0.9
    elif change == "feature_injection":
        rows[0]["views"]["history"]["actual_target_accuracy"] = 0.9
    elif change == "feature_nan":
        rows[0]["views"]["history"]["current.config.epochs"] = float("nan")
    elif change == "missing_feature":
        del rows[0]["views"]["history"]["current.config.epochs"]
    elif change == "wrong_delta":
        labels[rows[0]["example_id"]]["delta_accuracy"] += 0.5
    elif change == "duplicate_available":
        rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError):
        local.fit_predict(rows, labels, query, selection)


@pytest.mark.parametrize("change", [
    "too_few", "too_many", "duplicate_ids", "unknown_id", "few_sessions", "bad_model",
    "unknown_feature", "duplicate_feature", "too_many_features", "missing_parent_feature",
    "missing_weight", "zero_weight", "negative_weight", "nan_weight", "bool_weight", "large_weight",
    "empty_rationale", "extra_key", "wrong_order",
])
def test_invalid_selection_rejected(change):
    rows, labels, query, selection = fixture(18 if change == "too_many" else 12)
    if change == "too_few":
        selection["selected_ids"] = selection["selected_ids"][:7]
    elif change == "duplicate_ids":
        selection["selected_ids"][0] = selection["selected_ids"][1]
    elif change == "unknown_id":
        selection["selected_ids"][0] = "unknown"
    elif change == "few_sessions":
        for row in rows:
            row["cell_id"] = "one-session"
    elif change == "bad_model":
        selection["model"] = "unrestricted_python"
    elif change == "unknown_feature":
        selection["feature_keys"].append("example_id")
    elif change == "duplicate_feature":
        selection["feature_keys"].append("parent.accuracy")
    elif change == "too_many_features":
        selection["feature_keys"] = list(local.FEATURE_KEYS[:9])
    elif change == "missing_parent_feature":
        selection["feature_keys"] = ["current.config.epochs"]
    elif change == "missing_weight":
        selection["weights"].pop()
    elif change.endswith("_weight"):
        selection["weights"][0] = {
            "zero_weight": 0, "negative_weight": -1, "nan_weight": float("nan"),
            "bool_weight": True,
            "large_weight": 4.01,
        }[change]
    elif change == "empty_rationale":
        selection["rationale"] = " "
    elif change == "extra_key":
        selection["target_accuracy"] = 0.9
    elif change == "wrong_order":
        rows.reverse()
    with pytest.raises(ValueError):
        local.fit_predict(rows, labels, query, selection)


def test_group_weights_and_loso_never_split_session():
    rows, labels, query, selection = fixture()
    rows[0]["cell_id"] = rows[2]["cell_id"]
    selection["weights"] = [4.0] + [1.0] * (len(rows) - 1)
    result = local.fit_predict(rows, labels, query, selection)
    weights = result["fit_metadata"]["normalized_sample_weights"]
    sums = {}
    for row, weight in zip(rows, weights):
        sums[row["cell_id"]] = sums.get(row["cell_id"], 0) + weight
    expected = np.asarray([
        weight / sum(r["cell_id"] == row["cell_id"] for r in rows)
        for row, weight in zip(rows, selection["weights"])
    ])
    expected *= len(rows) / expected.sum()
    assert weights == pytest.approx(expected)
    assert sums[rows[0]["cell_id"]] > sums[rows[1]["cell_id"]]
    by_id = {row["example_id"]: row for row in rows}
    for fold in result["conditional_training_diagnostics"]["folds"]:
        training = {by_id[key]["cell_id"] for key in fold["fit_ids"]}
        validation = {by_id[key]["cell_id"] for key in fold["validation_ids"]}
        assert not training & validation
        assert validation == {fold["held_out_training_session"]}


def test_constant_training_feature_does_not_extrapolate_or_fit_query_scaler():
    rows, labels, query, selection = fixture()
    selection["feature_keys"] = ["parent.accuracy", "current.config.learning_rate"]
    baseline = local.fit_predict(rows, labels, query, selection)
    query["views"]["history"]["current.config.learning_rate"] = 100000.0
    result = local.fit_predict(rows, labels, query, selection)
    assert result["model_state"] == baseline["model_state"]
    assert result["predicted_accuracy"] == baseline["predicted_accuracy"]
    assert result["model_state"]["active"] == [True, False]
    assert result["model_state"]["coefficients"][1] == 0


def test_median_is_session_weighted_and_forecast_clipped():
    rows, labels, query, selection = fixture(8)
    selection.update(model="weighted_median", feature_keys=[])
    for row in rows:
        labels[row["example_id"]] = {"accuracy": 1.0, "delta_accuracy": 1.0 - row["parent_reference"]}
    query["parent_reference"] = 0.9
    query["views"]["history"]["parent.accuracy"] = 0.9
    result = local.fit_predict(rows, labels, query, selection)
    assert result["predicted_accuracy"] == 1.0
    assert result["predicted_delta"] == pytest.approx(0.1)
    assert result["raw_predicted_delta"] > result["predicted_delta"]
    assert result["clipped"]
    assert result["fit_metadata"]["feature_width"] == 0


def test_replay_tree_matches_sklearn_direct_prediction():
    from sklearn.tree import DecisionTreeRegressor

    rows, labels, query, selection = fixture()
    selection["model"] = "shallow_tree"
    result = local.fit_predict(rows, labels, query, selection)
    keys = selection["feature_keys"]
    x = [[row["views"]["history"][key] for key in keys] for row in rows]
    y = [labels[row["example_id"]]["delta_accuracy"] for row in rows]
    model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=3, random_state=local.SEED).fit(x, y)
    expected = model.predict([[query["views"]["history"][key] for key in keys]])[0]
    assert result["raw_predicted_delta"] == pytest.approx(expected)
    assert np.isfinite(result["model_state"]["value"]).all()
