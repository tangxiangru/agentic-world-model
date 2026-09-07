"""Synthetic checks for frozen clean-data model families, never real labels."""

import copy
import json

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.outcome_prediction.wm_clean_models import (
    NAMES,
    OLD_HGB_PARAMETERS,
    REGULARIZED_HGB_PARAMETERS,
    CleanPredictor,
)


def fixture_data(n=24):
    features = [{"x": i / n, "unused_missing": None} for i in range(n)]
    labels = [0.1 + 0.6 * i / n for i in range(n)]
    references = [0.1] * n
    groups = [f"session-{i // 6}" for i in range(n)]
    return features, labels, references, groups


@pytest.mark.parametrize("name", NAMES)
def test_fits_predicts_finite_clipped_vector_and_serializable_metadata(name):
    args = fixture_data()
    model = CleanPredictor(name).fit(*args)
    prediction = model.predict(args[0], args[2])
    assert prediction.shape == (len(args[0]),)
    assert np.isfinite(prediction).all()
    assert np.all((prediction >= 0) & (prediction <= 1))
    metadata = model.metadata()
    assert metadata["fitted"] is True
    assert metadata["train_rows"] == 24
    assert metadata["train_groups"] == 4
    assert metadata["reference_is_certified_parent_accuracy"] is False
    assert metadata["group_ids_used_as_features"] is False
    assert metadata["fit_seconds"] >= metadata["estimator_fit_seconds"] >= 0
    assert len(metadata["feature_vocabulary_sha256"]) == 64
    json.dumps(metadata, allow_nan=False)


def test_exact_frozen_legacy_hgb_parameters_and_seed_ensemble():
    model = CleanPredictor("old_delta").fit(*fixture_data())
    assert OLD_HGB_PARAMETERS == {
        "loss": "squared_error",
        "max_iter": 300,
        "learning_rate": 0.04,
        "max_leaf_nodes": 8,
        "min_samples_leaf": 5,
        "l2_regularization": 1.0,
        "max_depth": None,
        "early_stopping": False,
    }
    assert len(model.estimators_) == 3
    for seed, estimator in enumerate(model.estimators_):
        params = estimator.get_params()
        assert all(params[key] == value for key, value in OLD_HGB_PARAMETERS.items())
        assert params["random_state"] == seed
    assert model.metadata()["hgb_seeds"] == [0, 1, 2]


def test_fixed_regularized_hgb_and_ridge_parameters():
    regularized = CleanPredictor("regularized_delta").fit(*fixture_data())
    assert REGULARIZED_HGB_PARAMETERS == {
        "loss": "squared_error",
        "max_iter": 200,
        "learning_rate": 0.04,
        "max_leaf_nodes": 4,
        "min_samples_leaf": 10,
        "l2_regularization": 10.0,
        "max_depth": 2,
        "early_stopping": False,
    }
    assert len(regularized.estimators_) == 1
    assert regularized.estimators_[0].random_state == 0
    ridge = CleanPredictor("ridge_delta").fit(*fixture_data())
    assert ridge.estimators_[0].alpha == 100.0
    assert ridge.estimators_[0].solver == "svd"


@pytest.mark.parametrize("name", NAMES)
def test_unnormalized_equal_session_weights(name):
    features = [{"x": 0.0}] * 4
    labels = [0.0, 0.0, 0.0, 1.0]
    model = CleanPredictor(name).fit(features, labels, [0.0] * 4, ["large"] * 3 + ["small"])
    # One group at zero and one group at one: equal group mass gives 0.5,
    # not the 0.25 card-weighted mean. Total mass must remain TWO, not four.
    assert model.predict([{"x": 0.0}], [0.0])[0] == pytest.approx(0.5)
    meta = model.metadata()
    assert meta["weight_sum"] == pytest.approx(2)
    assert meta["weight_mean"] == pytest.approx(0.5)
    assert meta["weight_min"] == pytest.approx(1 / 3)
    assert meta["weight_max"] == 1


def test_legacy_delta_matches_direct_hgb_ensemble_with_original_weight_scale():
    features, y, references, groups = fixture_data()
    model = CleanPredictor("old_delta").fit(features, y, references, groups)
    matrix = np.asarray([[0.0, row["x"]] for row in features])
    weights = np.asarray([1 / groups.count(group) for group in groups])
    expected = (
        np.mean(
            [
                HistGradientBoostingRegressor(**OLD_HGB_PARAMETERS, random_state=seed)
                .fit(matrix, np.asarray(y) - references, sample_weight=weights)
                .predict(matrix)
                for seed in (0, 1, 2)
            ],
            axis=0,
        )
        + references
    )
    np.testing.assert_allclose(
        model.predict(features, references), np.clip(expected, 0, 1), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("name", ["old_delta", "regularized_delta", "ridge_delta"])
def test_delta_prediction_uses_caller_offset_and_clips(name):
    model = CleanPredictor(name).fit([{"x": 0.0}] * 4, [0.8] * 4, [0.3] * 4, ["s"] * 4)
    np.testing.assert_allclose(model.predict([{"x": 0.0}] * 3, [0.0, 0.2, 1.0]), [0.5, 0.7, 1.0])
    falling = CleanPredictor(name).fit([{"x": 0.0}] * 4, [0.1] * 4, [0.8] * 4, ["s"] * 4)
    np.testing.assert_allclose(falling.predict([{"x": 0.0}] * 2, [0.0, 1.0]), [0.0, 0.3])


def test_absolute_target_does_not_add_reference():
    model = CleanPredictor("old_absolute").fit([{"x": 0.0}] * 4, [0.4] * 4, [0.9] * 4, ["s"] * 4)
    np.testing.assert_allclose(model.predict([{"x": 0.0}] * 2, [0.0, 1.0]), [0.4, 0.4])
    assert model.metadata()["target"] == "child_accuracy"


@pytest.mark.parametrize("name", NAMES)
def test_vocabulary_preprocessing_and_metadata_are_train_only(name):
    features, labels, references, groups = fixture_data()
    model = CleanPredictor(name).fit(features, labels, references, groups)
    original_metadata = model.metadata()
    original_vocabulary = copy.deepcopy(model.vectorizer_.vocabulary_)
    test = [{"x": 0.3, "unseen_extreme_feature": 1e200, "unused_missing": 999}]
    model.predict(test, [0.1])
    assert model.vectorizer_.vocabulary_ == original_vocabulary
    assert "unseen_extreme_feature" not in model.vectorizer_.vocabulary_
    assert model.metadata() == original_metadata


@pytest.mark.parametrize("name", ["old_delta", "old_absolute", "regularized_delta"])
def test_old_hgb_native_missing_and_all_training_missing_zeroing(name):
    features, y, ref, groups = fixture_data()
    features[0]["x"] = None
    model = CleanPredictor(name).fit(features, y, ref, groups)
    assert model.imputer_ is None and model.scaler_ is None
    assert model.metadata()["model_feature_width"] == 2
    assert model.metadata()["all_training_missing_columns"] == 1
    # A column missing in ALL training rows stays zero even if test knows it.
    a = model.predict([{"x": 0.3, "unused_missing": None}], [0.1])
    b = model.predict([{"x": 0.3, "unused_missing": 1e6}], [0.1])
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(model.predict([{}], [0.1])).all()


def test_ridge_imputation_indicators_and_scaler_are_train_fitted():
    features = [{"x": 1.0, "z": None}, {"x": 3.0, "z": None}, {"x": None, "z": None}]
    model = CleanPredictor("ridge_delta").fit(features, [0.1, 0.2, 0.3], [0, 0, 0], ["s"] * 3)
    assert model.metadata()["model_feature_width"] == 4
    assert model.imputer_.statistics_[model.vectorizer_.vocabulary_["x"]] == 2.0
    statistics = model.imputer_.statistics_.copy()
    means = model.scaler_.mean_.copy()
    scales = model.scaler_.scale_.copy()
    model.predict([{"x": 1e6, "z": 123}, {}], [0, 0])
    np.testing.assert_array_equal(model.imputer_.statistics_, statistics)
    np.testing.assert_array_equal(model.scaler_.mean_, means)
    np.testing.assert_array_equal(model.scaler_.scale_, scales)


@pytest.mark.parametrize("name", NAMES)
def test_missing_key_and_explicit_none_have_same_meaning(name):
    model = CleanPredictor(name).fit(*fixture_data())
    np.testing.assert_array_equal(model.predict([{}], [0.1]), model.predict([{"x": None}], [0.1]))


@pytest.mark.parametrize(
    "key",
    ["example_id", "cell_id", "parent.id", "history.run_id", "session", "group", "candidate_id"],
)
def test_record_and_group_identity_feature_keys_are_rejected(key):
    with pytest.raises(ValueError):
        CleanPredictor("old_delta").fit([{key: 1.0}], [0], [0], ["s"])


@pytest.mark.parametrize(
    "value", ["0.1", "outcome prose", True, np.bool_(False), {}, [], float("inf"), float("-inf")]
)
def test_nonnumeric_boolean_and_infinite_features_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        CleanPredictor("old_delta").fit([{"x": value}], [0], [0], ["s"])


@pytest.mark.parametrize("field", ["labels", "references"])
@pytest.mark.parametrize(
    "value", [None, "0.2", True, np.bool_(True), float("nan"), float("inf"), -0.1, 1.1, [0.2]]
)
def test_invalid_label_and_reference_values_rejected(field, value):
    y, ref = ([value], [0.0]) if field == "labels" else ([0.0], [value])
    with pytest.raises((TypeError, ValueError)):
        CleanPredictor("old_delta").fit([{"x": 1.0}], y, ref, ["s"])


@pytest.mark.parametrize("position", [1, 2, 3])
def test_fit_shape_mismatch_rejected(position):
    args = list(fixture_data())
    args[position] = args[position][:-1]
    with pytest.raises(ValueError):
        CleanPredictor("old_delta").fit(*args)


@pytest.mark.parametrize("features", [[], [{}], {"x": 1.0}, [None], [{"": 1}], [{3: 1}]])
def test_empty_or_malformed_training_features_rejected(features):
    with pytest.raises((TypeError, ValueError)):
        CleanPredictor("old_delta").fit(features, [0], [0], ["s"])


@pytest.mark.parametrize("groups", ["session", [""], [None], [True], [123]])
def test_invalid_group_vector_rejected(groups):
    with pytest.raises((TypeError, ValueError)):
        CleanPredictor("old_delta").fit([{"x": 0}], [0], [0], groups)


def test_predict_requires_fit_and_matching_valid_references():
    with pytest.raises(ValueError):
        CleanPredictor("old_delta").predict([{"x": 0}], [0])
    model = CleanPredictor("old_delta").fit(*fixture_data())
    with pytest.raises(ValueError):
        model.predict([{"x": 0}], [])
    with pytest.raises((TypeError, ValueError)):
        model.predict([{"x": 0}], [True])
    with pytest.raises(ValueError):
        model.predict([{"example_id": 1}], [0])
    assert model.predict([], []).shape == (0,)


def test_unknown_model_rejected_and_unfitted_metadata_available():
    with pytest.raises(ValueError):
        CleanPredictor("huge_transformer")
    assert CleanPredictor("old_delta").metadata() == {"name": "old_delta", "fitted": False}


@pytest.mark.parametrize("name", NAMES)
def test_inputs_not_mutated_and_metadata_returns_deep_copy(name):
    args = fixture_data()
    original = copy.deepcopy(args)
    model = CleanPredictor(name).fit(*args)
    model.predict(args[0], args[2])
    assert args == original
    metadata = model.metadata()
    metadata["output_clip"][0] = -99
    assert model.metadata()["output_clip"] == [0.0, 1.0]


@pytest.mark.parametrize("name", NAMES)
def test_trusted_joblib_roundtrip_preserves_exact_predictions(tmp_path, name):
    args = fixture_data()
    model = CleanPredictor(name).fit(*args)
    path = tmp_path / "trusted_fixture.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(model.predict(args[0], args[2]), loaded.predict(args[0], args[2]))
    assert loaded.metadata() == model.metadata()


def test_failed_refit_does_not_publish_half_fitted_state():
    args = fixture_data()
    model = CleanPredictor("old_delta").fit(*args)
    before = model.predict(args[0], args[2])
    with pytest.raises((TypeError, ValueError)):
        model.fit([{"x": "bad"}], [0], [0], ["s"])
    np.testing.assert_array_equal(before, model.predict(args[0], args[2]))
