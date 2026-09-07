"""Synthetic broad-evidence regression tests; no real data, calls, or artifacts."""

import copy
import json
import warnings

import numpy as np
import pytest

from tools.outcome_prediction import wm_agent_broad_fit as broad
from tools.outcome_prediction import wm_agent_local_fit as old
from tools.outcome_prediction.wm_agent_recipe_data import FEATURE_KEYS as DATA_KEYS


def fixture(n=16):
    rows, labels = [], {}
    for i in range(n):
        reference = 0.2 + i / 200
        values = dict.fromkeys(broad.FEATURE_KEYS, 0.0)
        values.update({"parent.accuracy": reference, "parent.reference.measured": 1.0,
                       "current.config.epochs": float(i % 4), "current.config.learning_rate": 0.00002})
        values[DATA_KEYS[0]] = float(i % 2)
        key = f"synthetic-train-{i // 2}/exp-{i}"
        rows.append({"example_id": key, "cell_id": f"synthetic-train-{i // 2}",
                     "benchmark": "synthetic", "split": "train", "parent_reference": reference,
                     "reference_kind": "measured_parent", "views": {"history": values}})
        target = 0.05 + i / (n * 2)
        labels[key] = {"accuracy": target, "delta_accuracy": target - reference}
    query = copy.deepcopy(rows[0])
    query.update(example_id="synthetic-heldout/exp-0", cell_id="synthetic-heldout", split="test")
    spec = {"weights": [1.0] * n, "target": "delta", "model": "ridge_10",
            "feature_keys": ["parent.accuracy", "current.config.epochs", DATA_KEYS[0]],
            "rationale": "Synthetic full-pool test specification."}
    return rows, labels, query, spec


def test_exact_augmented_schema_without_frozen_global_mutation():
    assert len(DATA_KEYS) == 57
    assert len(broad.FEATURE_KEYS) == 208
    assert set(broad.FEATURE_KEYS) == set(old.FEATURE_KEYS) | set(DATA_KEYS)
    assert not set(old.FEATURE_KEYS) & set(DATA_KEYS)
    assert len(old.FEATURE_KEYS) == 151
    rows, _, query, spec = fixture()
    broad.validate_spec(spec, rows, query)
    assert len(old.FEATURE_KEYS) == 151


@pytest.mark.parametrize("model", broad.MODELS)
@pytest.mark.parametrize("target", broad.TARGETS)
def test_both_targets_all_models_deterministic_and_json_replay(model, target):
    rows, labels, query, spec = fixture()
    spec.update(model=model, target=target)
    spec["weights"] = [0.5, 1, 1.5, 2] * 4
    if model == "weighted_median":
        spec["feature_keys"] = []
    before = copy.deepcopy((rows, labels, query, spec))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = broad.fit_predict(rows, labels, query, spec)
    assert result == broad.fit_predict(rows, labels, query, spec)
    saved = json.loads(json.dumps(result, allow_nan=False))
    replay = broad.predict_from_state(saved["model_state"], query)
    assert all(result[key] == value for key, value in replay.items())
    assert 0 <= result["predicted_accuracy"] <= 1
    assert result["predicted_delta"] == result["predicted_accuracy"] - query["parent_reference"]
    assert result["fit_metadata"]["training_examples"] == len(rows)
    assert result["fit_metadata"]["training_ids"] == [row["example_id"] for row in rows]
    assert result["fit_metadata"]["training_equal_session_mae"] >= 0
    assert result["conditional_training_diagnostics"]["equal_session_mae"] >= 0
    assert before == (rows, labels, query, spec)


def test_median_accuracy_and_delta_add_parent_only_for_delta():
    rows, labels, query, spec = fixture()
    for row in rows:
        labels[row["example_id"]] = {"accuracy": 0.0, "delta_accuracy": -row["parent_reference"]}
    spec.update(model="weighted_median", feature_keys=[], target="accuracy")
    query["parent_reference"] = 0.9
    query["views"]["history"]["parent.accuracy"] = 0.9
    absolute = broad.fit_predict(rows, labels, query, spec)
    delta = broad.fit_predict(rows, labels, query, {**spec, "target": "delta"})
    assert absolute["predicted_accuracy"] == 0
    assert absolute["predicted_delta"] == -0.9
    assert absolute["raw_prediction"] == 0
    assert delta["predicted_accuracy"] > 0.6
    assert delta["raw_prediction"] < 0
    assert delta["unclipped_predicted_accuracy"] == query["parent_reference"] + delta["raw_prediction"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1, True, "0.2"])
@pytest.mark.parametrize("owner", ["target", "parent", "query"])
@pytest.mark.parametrize("mode", broad.TARGETS)
def test_missing_or_invalid_accuracy_rejected_for_both_targets(value, owner, mode):
    rows, labels, query, spec = fixture()
    spec["target"] = mode
    if owner == "target":
        labels[rows[0]["example_id"]]["accuracy"] = value
    elif owner == "parent":
        rows[0]["parent_reference"] = value
    else:
        query["parent_reference"] = value
    with pytest.raises(ValueError, match="Accuracy"):
        broad.fit_predict(rows, labels, query, spec)


def test_valid_zero_parent_and_target_are_kept_in_training():
    rows, labels, query, spec = fixture()
    rows[0]["parent_reference"] = 0.0
    rows[0]["views"]["history"]["parent.accuracy"] = 0.0
    labels[rows[0]["example_id"]] = {"accuracy": 0.0, "delta_accuracy": 0.0}
    result = broad.fit_predict(rows, labels, query, spec)
    assert rows[0]["example_id"] in result["fit_metadata"]["training_ids"]
    assert result["fit_metadata"]["normalized_sample_weights"][0] > 0


@pytest.mark.parametrize("invalid", [
    "row_label", "query_label", "extra_label", "missing_label", "missing_delta", "wrong_delta",
    "heldout_row", "overlap", "benchmark", "duplicate", "unknown_reference", "reference_feature",
    "reference_flag", "missing_data_feature", "missing_old_feature", "extra_feature", "nan_data",
    "boolean_data", "unknown_view", "data_in_parent_view",
])
def test_boundaries_fail_closed(invalid):
    rows, labels, query, spec = fixture()
    if invalid == "row_label":
        rows[0]["accuracy"] = 0.9
    elif invalid == "query_label":
        query["accuracy"] = 0.9
    elif invalid == "extra_label":
        labels[query["example_id"]] = {"accuracy": 0.9, "delta_accuracy": 0.7}
    elif invalid == "missing_label":
        labels.pop(rows[0]["example_id"])
    elif invalid == "missing_delta":
        labels[rows[0]["example_id"]].pop("delta_accuracy")
    elif invalid == "wrong_delta":
        labels[rows[0]["example_id"]]["delta_accuracy"] = 1.0
    elif invalid == "heldout_row":
        rows[0]["split"] = "test"
    elif invalid == "overlap":
        query["cell_id"] = rows[0]["cell_id"]
    elif invalid == "benchmark":
        rows[0]["benchmark"] = "other"
    elif invalid == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif invalid == "unknown_reference":
        rows[0]["reference_kind"] = "imputed"
    elif invalid == "reference_feature":
        rows[0]["views"]["history"]["parent.accuracy"] = 0.9
    elif invalid == "reference_flag":
        rows[0]["views"]["history"]["parent.reference.measured"] = 0.0
    elif invalid == "missing_data_feature":
        rows[0]["views"]["history"].pop(DATA_KEYS[0])
    elif invalid == "missing_old_feature":
        rows[0]["views"]["history"].pop("current.config.epochs")
    elif invalid == "extra_feature":
        rows[0]["views"]["history"]["future_accuracy"] = 0.9
    elif invalid == "nan_data":
        rows[0]["views"]["history"][DATA_KEYS[0]] = float("nan")
    elif invalid == "boolean_data":
        rows[0]["views"]["history"][DATA_KEYS[0]] = True
    elif invalid == "unknown_view":
        rows[0]["views"]["outcomes"] = {}
    else:
        rows[0]["views"]["parent"] = {DATA_KEYS[0]: 0.0}
    with pytest.raises(ValueError):
        broad.fit_predict(rows, labels, query, spec)


@pytest.mark.parametrize("invalid", [
    "selected_ids", "missing_target", "bad_target", "bad_model", "bad_feature", "duplicate_feature",
    "too_many_features", "no_parent", "empty_rationale", "short_weights", "no_weights", "few_sessions",
])
def test_invalid_spec_rejected(invalid):
    rows, labels, query, spec = fixture()
    if invalid == "selected_ids":
        spec["selected_ids"] = [rows[0]["example_id"]]
    elif invalid == "missing_target":
        spec.pop("target")
    elif invalid == "bad_target":
        spec["target"] = "guaranteed_improvement"
    elif invalid == "bad_model":
        spec["model"] = "execute_code"
    elif invalid == "bad_feature":
        spec["feature_keys"].append("example_id")
    elif invalid == "duplicate_feature":
        spec["feature_keys"].append("parent.accuracy")
    elif invalid == "too_many_features":
        spec["feature_keys"] = list(broad.FEATURE_KEYS[:13])
    elif invalid == "no_parent":
        spec["feature_keys"] = [DATA_KEYS[0]]
    elif invalid == "empty_rationale":
        spec["rationale"] = " "
    elif invalid == "short_weights":
        spec["weights"].pop()
    elif invalid == "no_weights":
        spec["weights"] = None
    else:
        for row in rows:
            row["cell_id"] = "single-session"
    with pytest.raises(ValueError):
        broad.fit_predict(rows, labels, query, spec)


@pytest.mark.parametrize("weight", [None, True, "1", -1, 0, 0.49, 2.01, float("nan"), float("inf")])
def test_weight_bounds_prevent_discarding_any_row(weight):
    rows, labels, query, spec = fixture()
    spec["weights"][0] = weight
    with pytest.raises(ValueError):
        broad.fit_predict(rows, labels, query, spec)


@pytest.mark.parametrize("benchmark,count", broad.EXPECTED_TRAIN_COUNTS.items())
def test_known_benchmark_requires_full_frozen_cohort(benchmark, count):
    rows, _, query, spec = fixture(count)
    query["benchmark"] = benchmark
    for row in rows:
        row["benchmark"] = benchmark
    broad.validate_spec(spec, rows, query)
    with pytest.raises(ValueError, match="subset selection forbidden"):
        broad.validate_spec({**spec, "weights": spec["weights"][:-1]}, rows[:-1], query)


def test_mixed_weights_and_ess_are_exact_and_all_rows_retained():
    rows, labels, query, spec = fixture()
    rows[0]["cell_id"] = rows[2]["cell_id"]
    spec["weights"] = [2.0] * 4 + [0.5] * 12
    result = broad.fit_predict(rows, labels, query, spec)
    meta = result["fit_metadata"]
    counts = {row["cell_id"]: sum(r["cell_id"] == row["cell_id"] for r in rows) for row in rows}
    uniform = np.asarray([1.0 / counts[row["cell_id"]] for row in rows])
    adjusted = np.asarray([weight / counts[row["cell_id"]] for row, weight in zip(rows, spec["weights"])])
    uniform *= len(rows) / uniform.sum()
    adjusted *= len(rows) / adjusted.sum()
    mixed = 0.5 * uniform + 0.5 * adjusted
    assert meta["normalized_sample_weights"] == pytest.approx(mixed)
    assert meta["session_balanced_component"] == pytest.approx(uniform)
    assert meta["normalized_relevance_component"] == pytest.approx(adjusted)
    assert mixed.mean() == pytest.approx(1)
    assert (mixed >= 0.5 * uniform).all()
    assert meta["row_effective_sample_size"] == pytest.approx(mixed.sum() ** 2 / (mixed @ mixed))
    session_weights = np.asarray(list(meta["session_totals"].values()))
    assert meta["session_effective_sample_size"] == pytest.approx(session_weights.sum() ** 2 / (session_weights @ session_weights))
    for fold in result["conditional_training_diagnostics"]["folds"]:
        assert len(fold["fit_ids"]) + len(fold["validation_ids"]) == len(rows)
        by_id = {row["example_id"]: row for row in rows}
        assert not ({by_id[key]["cell_id"] for key in fold["fit_ids"]}
                    & {by_id[key]["cell_id"] for key in fold["validation_ids"]})


def test_uniform_relevance_is_exactly_session_balanced():
    rows, _, _, _ = fixture()
    rows[0]["cell_id"] = rows[2]["cell_id"]
    mixed, uniform, adjusted = broad.effective_weights(rows, [1.0] * len(rows))
    assert mixed.tolist() == uniform.tolist() == adjusted.tolist()
    for session in {row["cell_id"] for row in rows}:
        total = sum(float(weight) for row, weight in zip(rows, mixed) if row["cell_id"] == session)
        assert total == pytest.approx(len(rows) / len({row["cell_id"] for row in rows}))


def test_scaling_constant_columns_warning_free_and_train_only():
    rows, labels, query, spec = fixture()
    spec["feature_keys"] = ["parent.accuracy", "current.config.learning_rate"]
    spec["weights"] = [0.5, 1, 1.5, 2] * 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        original = broad.fit_predict(rows, labels, query, spec)
        query["views"]["history"]["current.config.learning_rate"] = 1e100
        changed = broad.fit_predict(rows, labels, query, spec)
    assert original["model_state"] == changed["model_state"]
    assert original["predicted_accuracy"] == changed["predicted_accuracy"]
    assert original["model_state"]["scaler"]["active"] == [True, False]
    assert original["model_state"]["coefficients"][1] == 0


def test_weighted_scaling_matches_train_weighted_mean_variance():
    x = np.asarray([[1e-5, 10.0, 7.0], [2e-5, 20.0, 7.0], [4e-5, -3.0, 7.0]])
    weights = np.asarray([0.5, 1, 2.0])
    transformed, scaler = broad._weighted_scaling(x, weights)
    for column in (0, 1):
        assert np.average(transformed[:, column], weights=weights) == pytest.approx(0.0, abs=1e-14)
        assert np.average(transformed[:, column] ** 2, weights=weights) == pytest.approx(1.0)
    assert transformed[:, 2].tolist() == [0.0, 0.0, 0.0]
    assert scaler["active"] == [True, True, False]


def test_shallow_tree_min_leaf_and_direct_sklearn_replay():
    from sklearn.tree import DecisionTreeRegressor

    rows, labels, query, spec = fixture()
    spec.update(model="shallow_tree", target="accuracy")
    result = broad.fit_predict(rows, labels, query, spec)
    weights, _, _ = broad.effective_weights(rows, spec["weights"])
    keys = spec["feature_keys"]
    x = [[row["views"]["history"][key] for key in keys] for row in rows]
    y = [labels[row["example_id"]]["accuracy"] for row in rows]
    model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, random_state=broad.SEED).fit(x, y, sample_weight=weights)
    expected = model.predict([[query["views"]["history"][key] for key in keys]])[0]
    assert result["raw_prediction"] == pytest.approx(expected)
    assert result["model_state"]["min_samples_leaf"] == 5
    for i, left in enumerate(result["model_state"]["children_left"]):
        if left == -1:
            assert result["model_state"]["node_samples"][i] >= 5


def test_declining_forecast_not_forced_to_improve_parent():
    rows, labels, query, spec = fixture()
    for row in rows:
        labels[row["example_id"]] = {"accuracy": 0.0, "delta_accuracy": -row["parent_reference"]}
    result = broad.fit_predict(rows, labels, query, {**spec, "target": "accuracy"})
    assert result["predicted_accuracy"] == 0.0
    assert result["predicted_delta"] < 0
