import copy
import json

import numpy as np
import pytest

from tools.outcome_prediction import wm_delta_model as wm


def row(x=1.0, *, parent=0.5, cell="cell-a", history=None, benchmark="synthetic"):
    return {
        "example_id": "example-a",
        "cell_id": cell,
        "benchmark": benchmark,
        "parent_key": "parent-a",
        "parent_accuracy": parent,
        "current_features": {"lr": x},
        "history": [] if history is None else history,
    }


def history(value, accuracy=None, relation="weights"):
    return {"features": {"lr": value}, "accuracy": accuracy, "relation": relation}


@pytest.mark.parametrize("name", sorted(wm.NAMES))
def test_signed_deltas_and_single_row_fit(name):
    model = wm.LightweightDeltaModel(name).fit([row(parent=0.8)], [0.2])
    delta = model.predict_delta([row(parent=0.8)])
    assert delta.shape == (1,)
    assert delta[0] == pytest.approx(0.0 if name == "carry_parent" else -0.6)
    assert model.predict_accuracy([row(parent=0.8)])[0] == pytest.approx(
        0.8 if name == "carry_parent" else 0.2
    )
    assert model.predict_delta([]).shape == (0,)
    assert model.predict_accuracy([]).shape == (0,)
    json.dumps(model.metadata(), allow_nan=False)


@pytest.mark.parametrize("name", sorted(wm.NAMES))
def test_same_parent_ranking_algebra_and_carry_ties(name):
    rows = [row(float(i), cell=f"cell-{i % 3}") for i in range(12)]
    targets = np.linspace(0.2, 0.8, len(rows))
    model = wm.LightweightDeltaModel(name).fit(rows, targets)
    delta, accuracy = model.predict_delta(rows), model.predict_accuracy(rows)
    np.testing.assert_allclose(accuracy, np.clip(0.5 + delta, 0, 1))
    if name == "carry_parent":
        assert len(set(accuracy)) == 1
    else:
        assert accuracy[-1] > accuracy[0]
        assert np.array_equal(np.argsort(delta), np.argsort(accuracy))
    other_identity = copy.deepcopy(rows)
    for i, candidate in enumerate(other_identity):
        candidate["parent_key"] = f"different-parent-{i}"
        candidate["example_id"] = f"different-example-{i}"
        candidate["cell_id"] = f"different-cell-{i}"
    # Identity is not a feature. This does NOT declare these different parents
    # to be a valid same-parent choice group; the dataset builder owns grouping.
    np.testing.assert_array_equal(delta, model.predict_delta(other_identity))


class GuardedRow(dict):
    def __getitem__(self, key):
        if key in {"target", "label", "child_accuracy", "official_score"}:
            raise AssertionError("Child-label field was accessed")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in {"target", "label", "child_accuracy", "official_score"}:
            raise AssertionError("Child-label field was accessed")
        return super().get(key, default)

    def items(self):
        raise AssertionError("Whole row was scanned")

    def __iter__(self):
        raise AssertionError("Whole row was scanned")


@pytest.mark.parametrize("name", sorted(wm.NAMES))
def test_target_fields_ignored_and_no_whole_row_scan(name):
    guarded = GuardedRow(row())
    for key in ("target", "label", "child_accuracy", "official_score"):
        guarded[key] = object()
    model = wm.LightweightDeltaModel(name).fit([guarded], [0.3])
    assert np.isfinite(model.predict_delta([guarded])).all()
    assert np.isfinite(model.predict_accuracy([guarded])).all()


def test_fixed_size_history_aggregation_and_missing_score_semantics():
    candidate = row(
        7.0,
        history=[
            history(1.0, 0.0),
            {"features": {}, "accuracy": None, "relation": "data"},
            history(5.0, 0.4),
        ],
    )
    checked, _ = wm._rows([candidate])
    features = wm._features(checked[0], history=True)
    assert features["history_count"] == 3
    assert features["history_known_accuracy_count"] == 2
    assert features["history_accuracy_mean"] == pytest.approx(0.2)
    assert features["history_accuracy_recent"] == pytest.approx(0.4)
    assert features["history_last_gain"] == pytest.approx(0.4)
    assert features[wm._column("history_relation_count", "weights")] == 2
    assert features[wm._column("history_sum", "lr")] == 6
    assert features[wm._column("history_mean", "lr")] == 3
    assert features[wm._column("history_recent", "lr")] == 5
    assert features[wm._column("history_recency_mean", "lr")] == pytest.approx(4.2)
    assert features[wm._column("current_minus_history_mean", "lr")] == 4
    assert features[wm._column("current_minus_history_recent", "lr")] == 2
    assert features[wm._column("current_times_history_recent", "lr")] == 35
    longer = copy.deepcopy(candidate)
    longer["history"] *= 100
    longer_checked, _ = wm._rows([longer])
    assert set(wm._features(longer_checked[0], history=True)) == set(features)
    model = wm.LightweightDeltaModel("ridge_history_delta").fit([candidate], [0.7])
    width = model.metadata()["feature_count"]
    assert model.predict_delta([longer]).shape == (1,)
    assert model.metadata()["feature_count"] == width
    assert not any("step_index" in name for name in model.feature_names_)


def test_recency_mean_stable_for_long_missing_suffix_and_last_gain_can_be_negative():
    entries = [history(2.0, 0.8), history(4.0, 0.2)]
    entries += [{"features": {}, "accuracy": None, "relation": "data"}] * 2000
    checked, _ = wm._rows([row(history=entries)])
    f = wm._features(checked[0], history=True)
    assert f[wm._column("history_recency_mean", "lr")] == pytest.approx(10 / 3)
    assert f["history_last_gain"] == pytest.approx(-0.6)


def test_train_only_vocabulary_scaler_missing_and_unseen_features():
    a, b = row(0.0), row(2.0)
    model = wm.LightweightDeltaModel("ridge_current_delta").fit([a, b], [0.3, 0.7])
    vocabulary = copy.deepcopy(model.vectorizer_.vocabulary_)
    means = model.scaler_.mean_.copy()
    unseen = row(1000000.0, history=[history(1.0, 0.99, relation="unseen_relation")])
    unseen["current_features"]["never_train"] = 123.0
    missing = row()
    missing["current_features"] = {}
    model.predict_delta([unseen, missing])
    assert model.vectorizer_.vocabulary_ == vocabulary
    np.testing.assert_array_equal(model.scaler_.mean_, means)
    assert not any("never_train" in k or "history" in k for k in model.feature_names_)
    checked, _ = wm._rows([a, missing])
    matrix = model.vectorizer_.transform([wm._features(x, history=False) for x in checked])
    assert not np.array_equal(matrix[0], matrix[1])  # zero has presence, missing does not


@pytest.mark.parametrize(
    "name", ["ridge_history_delta", "trees_history_delta", "ridge_history_absolute"]
)
def test_history_vocabulary_frozen_and_repeatable(name):
    rows = [row(float(i), history=[history(float(i), 0.2)]) for i in range(6)]
    labels = np.linspace(0.1, 0.7, 6)
    first = wm.LightweightDeltaModel(name).fit(rows, labels)
    second = wm.LightweightDeltaModel(name).fit(rows, labels)
    np.testing.assert_array_equal(first.predict_delta(rows), second.predict_delta(rows))
    names = first.feature_names_.copy()
    query = row(history=[history(0.0, 0.0)] * 100)
    query["history"].append(
        {"features": {"test_only_key": 999.0}, "accuracy": None, "relation": "test_only_relation"}
    )
    before = copy.deepcopy(query)
    assert np.isfinite(first.predict_delta([query])).all()
    assert query == before
    assert first.feature_names_ == names
    assert not any("test_only" in key for key in first.vectorizer_.vocabulary_)


def test_equal_cell_total_weights_and_weighted_scaler():
    rows = [row(0.0, cell="many"), row(0.0, cell="many"), row(9.0, cell="one")]
    before = copy.deepcopy(rows)
    model = wm.LightweightDeltaModel("ridge_current_delta").fit(rows, [0.1, 0.2, 0.9])
    assert rows == before
    np.testing.assert_allclose(model.sample_weight_, [0.75, 0.75, 1.5])
    column = model.vectorizer_.vocabulary_[wm._column("current_value", "lr")]
    assert model.scaler_.mean_[column] == pytest.approx(4.5)
    assert model.metadata()["training_cells"] == 2


@pytest.mark.parametrize("name", sorted(wm.NAMES - {"carry_parent"}))
def test_raw_signed_output_not_clipped_but_accuracy_is(name):
    model = wm.LightweightDeltaModel(name).fit([row()], [0.5])
    model.estimator_.predict = lambda x: np.asarray([-2.0, 2.0])
    candidates = [row(parent=0.2), row(parent=0.8)]
    expected = [-2.2, 1.2] if name.endswith("absolute") else [-2.0, 2.0]
    np.testing.assert_allclose(model.predict_delta(candidates), expected)
    np.testing.assert_array_equal(model.predict_accuracy(candidates), [0.0, 1.0])


@pytest.mark.parametrize("name", sorted(wm.NAMES))
def test_parent_zero_valid_missing_or_nonfinite_rejected(name):
    model = wm.LightweightDeltaModel(name).fit([row(parent=0.0)], [0.0])
    assert model.predict_accuracy([row(parent=0.0)])[0] == 0
    for bad in (None, float("nan"), float("inf"), -0.1, 1.1, True):
        with pytest.raises((ValueError, TypeError)):
            model.predict_delta([row(parent=bad)])
    missing = row()
    del missing["parent_accuracy"]
    with pytest.raises(ValueError):
        model.predict_delta([missing])


@pytest.mark.parametrize("bad", [None, True, float("inf"), float("nan"), "1"])
def test_bad_current_or_history_features_rejected(bad):
    candidate = row()
    candidate["current_features"]["lr"] = bad
    with pytest.raises((ValueError, TypeError)):
        wm.LightweightDeltaModel("ridge_current_delta").fit([candidate], [0.5])
    with pytest.raises((ValueError, TypeError)):
        wm.LightweightDeltaModel("ridge_history_delta").fit([row(history=[history(bad)])], [0.5])


def test_input_validation_and_frozen_parameters():
    with pytest.raises(ValueError):
        wm.LightweightDeltaModel("unknown")
    model = wm.LightweightDeltaModel("ridge_current_delta")
    with pytest.raises(RuntimeError):
        model.predict_delta([row()])
    with pytest.raises(RuntimeError):
        model.metadata()
    for rows, targets in (
        ([], []),
        ([row()], []),
        ([row()], [float("nan")]),
        ([row(), row(benchmark="other")], [0.2, 0.3]),
    ):
        with pytest.raises(ValueError):
            model.fit(rows, targets)
    model.fit([row()], [0.4])
    with pytest.raises(ValueError):
        model.predict_delta([row(benchmark="other")])
    assert model.metadata()["estimator_parameters"]["alpha"] == 10
    trees = wm.LightweightDeltaModel("trees_history_delta").fit([row()], [0.4])
    params = trees.metadata()["estimator_parameters"]
    assert {
        k: params[k]
        for k in ("n_estimators", "max_depth", "min_samples_leaf", "n_jobs", "random_state")
    } == {
        "n_estimators": 64,
        "max_depth": 3,
        "min_samples_leaf": 3,
        "n_jobs": 1,
        "random_state": 20260905,
    }
