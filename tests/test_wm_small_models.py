"""Synthetic-only tests for the low-sample predictor interface and fit boundaries."""

import copy
import json
import pickle

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from tools.outcome_prediction.wm_small_models import HGB_PARAMETERS, NAMES, SmallPredictor


def fixture(n=18):
    rows = [
        {
            "lr": 1e-4 * (i + 1),
            "epochs": None if i % 4 == 0 else i % 3 + 1,
            "family": "sft" if i % 2 else "rft",
            "bf16": True,
        }
        for i in range(n)
    ]
    reference = np.linspace(0.1, 0.35, n)
    y = np.clip(reference + np.linspace(-0.08, 0.25, n), 0, 1)
    groups = ["session-a"] * (n // 2) + ["session-b"] * (n - n // 2)
    embedding = np.random.default_rng(42).normal(size=(n, 24))
    return rows, y, reference, groups, embedding


def fitted(name="ridge_delta", n=18, alpha=10):
    rows, y, reference, groups, embedding = fixture(n)
    model = SmallPredictor(name, alpha=alpha)
    return model.fit(
        rows, y, reference, groups, embedding if name.startswith("embedding_") else None
    )


@pytest.mark.parametrize("name", NAMES)
def test_all_variants_fit_predict_cpu_and_metadata_is_json_safe(name):
    rows, y, reference, groups, embeddings = fixture()
    before = copy.deepcopy(rows)
    model = SmallPredictor(name)
    assert model.metadata()["fitted"] is False
    assert (
        model.fit(rows, y, reference, groups, embeddings if name.startswith("embedding_") else None)
        is model
    )
    prediction = model.predict(
        rows, reference, embeddings if name.startswith("embedding_") else None
    )
    assert prediction.shape == y.shape and np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    assert rows == before
    metadata = model.metadata()
    json.dumps(metadata, allow_nan=False)
    assert metadata["train_rows"] == len(rows) and metadata["train_groups"] == 2
    assert metadata["fit_seconds"] >= metadata["estimator_fit_seconds"] >= 0
    assert metadata["model_feature_width"] > 0
    assert metadata["sample_weight_mean"] == pytest.approx(1)
    assert metadata["group_identity_used_as_feature"] is False
    assert "session-a" not in json.dumps(metadata)
    metadata["train_rows"] = -1
    assert model.metadata()["train_rows"] == len(rows)


@pytest.mark.parametrize("name", NAMES)
def test_trusted_local_pickle_roundtrip_preserves_predictions_and_metadata(name):
    rows, _, reference, _, embeddings = fixture()
    model = fitted(name)
    loaded = pickle.loads(pickle.dumps(model))
    kwargs = {"embeddings": embeddings} if name.startswith("embedding_") else {}
    np.testing.assert_array_equal(
        model.predict(rows, reference, **kwargs), loaded.predict(rows, reference, **kwargs)
    )
    assert model.metadata() == loaded.metadata()


@pytest.mark.parametrize("name", ["ridge_delta", "ridge_absolute", "hgb_delta", "hgb_absolute"])
def test_equal_total_session_weights_and_target_parameterization(name):
    rows = [{"constant": 1}] * 4
    y = [0.2, 0.2, 0.2, 0.8]
    reference = [0.1] * 4
    groups = ["large", "large", "large", "small"]
    model = SmallPredictor(name).fit(rows, y, reference, groups)
    np.testing.assert_allclose(model.predict(rows, reference), 0.5, atol=1e-10)
    different_parent = model.predict(rows, [0.2] * 4)
    np.testing.assert_allclose(different_parent, 0.6 if name.endswith("_delta") else 0.5)
    assert model.metadata()["sample_weight_min"] == pytest.approx(2 / 3)
    assert model.metadata()["sample_weight_max"] == pytest.approx(2)


def test_ridge_scaler_is_fitted_with_equal_group_weights():
    rows = [{"x": 0.0}, {"x": 0.0}, {"x": 0.0}, {"x": 10.0}]
    model = SmallPredictor("ridge_delta").fit(rows, [0.2] * 4, [0.1] * 4, ["a", "a", "a", "b"])
    index = model.vectorizer_.vocabulary_["value\x1fx"]
    assert model.scaler_.mean_[index] == pytest.approx(5)
    assert model.scaler_.scale_[index] == pytest.approx(5)


def test_missing_values_differ_from_explicit_zero_and_future_missingness_is_represented():
    rows = [{"x": 2, "always": 4, "all_missing": None}, {"x": 8, "always": 8}, {}]
    model = SmallPredictor("ridge_absolute").fit(rows, [0.2, 0.4, 0.6], [0.1] * 3, ["a", "b", "c"])
    assert model.imputer_.statistics_[model.vectorizer_.vocabulary_["value\x1fx"]] == 5
    vocab = model.vectorizer_.vocabulary_
    assert "missing\x1fx" in vocab and "missing\x1fall_missing" in vocab
    assert model.imputer_.statistics_[vocab["value\x1fall_missing"]] == 0
    model.predict([{}, {"x": 0}, {"x": None}, {"x": np.nan}], [0.1] * 4)
    assert model.metadata()["raw_feature_keys"] == 3
    complete = SmallPredictor("ridge_absolute").fit(
        [{"x": 2}, {"x": 8}], [0.2, 0.8], [0.1, 0.1], ["a", "b"]
    )
    assert "missing\x1fx" in complete.vectorizer_.vocabulary_
    assert complete.predict([{}], [0.1]).shape == (1,)


def test_unseen_test_features_categories_and_extreme_values_do_not_refit_preprocessors():
    model = fitted("embedding_ridge_delta")
    before = pickle.dumps((model.vectorizer_, model.imputer_, model.scaler_, model.pca_))
    metadata = model.metadata()
    model.predict(
        [{"lr": 100.0, "family": "never-seen", "test_only": 999}], [0.3], np.full((1, 24), 500.0)
    )
    after = pickle.dumps((model.vectorizer_, model.imputer_, model.scaler_, model.pca_))
    assert before == after and model.metadata() == metadata
    assert not any(
        "test_only" in key or "never-seen" in key for key in model.vectorizer_.vocabulary_
    )


@pytest.mark.parametrize("n,width,expected", [(2, 8, 1), (6, 3, 3), (8, 20, 7), (20, 40, 16)])
def test_embedding_pca_is_train_only_capped_and_concatenated(n, width, expected):
    rows = [{"x": i} for i in range(n)]
    embeddings = np.random.default_rng(13).normal(size=(n, width))
    model = SmallPredictor("embedding_ridge_delta").fit(
        rows, [0.5] * n, [0.2] * n, list(range(n)), embeddings
    )
    assert model.pca_.n_components_ == expected
    np.testing.assert_allclose(model.pca_.mean_, embeddings.mean(axis=0))
    assert model.metadata()["model_feature_width"] == model.structured_width_ + expected
    assert model.metadata()["pca_weighting"] == "unweighted_train_rows"


def test_embedding_only_input_is_supported_without_invented_structured_features():
    embeddings = np.random.default_rng(1).normal(size=(5, 8))
    model = SmallPredictor("embedding_kernel_delta").fit(
        [{}] * 5, [0.5] * 5, [0.2] * 5, list(range(5)), embeddings
    )
    assert model.metadata()["structured_feature_width"] == 0
    assert model.metadata()["model_feature_width"] == 4
    assert model.predict([{}], [0.3], embeddings[:1]).shape == (1,)


def test_estimator_parameters_are_fixed_and_alpha_is_passed_through():
    for name in ("hgb_delta", "hgb_absolute"):
        model = fitted(name, alpha=100)
        assert isinstance(model.estimator_, HistGradientBoostingRegressor)
        params = model.estimator_.get_params()
        assert all(params[key] == value for key, value in HGB_PARAMETERS.items())
        assert model.metadata()["alpha_used"] is False
    ridge = fitted("embedding_ridge_delta", alpha=100)
    assert isinstance(ridge.estimator_, Ridge) and ridge.estimator_.alpha == 100
    kernel = fitted("embedding_kernel_delta", alpha=100)
    assert isinstance(kernel.estimator_, KernelRidge)
    assert kernel.estimator_.alpha == 100 and kernel.estimator_.kernel == "rbf"
    assert kernel.estimator_.gamma == 1 / kernel.metadata()["model_feature_width"]


def test_delta_predictions_are_clipped_after_reference_is_added():
    model = SmallPredictor("ridge_delta").fit([{"x": 1}] * 2, [1, 1], [0, 0], ["a", "b"])
    np.testing.assert_array_equal(model.predict([{"x": 1}], [1]), [1])
    model = SmallPredictor("ridge_delta").fit([{"x": 1}] * 2, [0, 0], [1, 1], ["a", "b"])
    np.testing.assert_array_equal(model.predict([{"x": 1}], [0]), [0])


@pytest.mark.parametrize("name", ["invalid", "ridge", "embedding_hgb_delta"])
def test_unknown_names_rejected(name):
    with pytest.raises(ValueError):
        SmallPredictor(name)


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf"), True, "10", 10**1000])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises((ValueError, TypeError)):
        SmallPredictor("ridge_delta", alpha=alpha)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "1"])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError):
        SmallPredictor("ridge_delta", seed=seed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01, None, True, [0.2]])
@pytest.mark.parametrize("target", ["child", "parent"])
def test_invalid_accuracy_or_reference_rejected(value, target):
    y, ref = [0.2, 0.3], [0.1, 0.1]
    (y if target == "child" else ref)[0] = value
    with pytest.raises((ValueError, TypeError)):
        SmallPredictor("ridge_delta").fit([{"x": 1}, {"x": 2}], y, ref, ["a", "b"])


@pytest.mark.parametrize(
    "key",
    [
        "id",
        "example_id",
        "cell_id",
        "run_id",
        "session_id",
        "group_id",
        "history.parent_id",
        "history_parent_id",
        "record_id",
        "candidate_id",
        "groups",
    ],
)
def test_identity_features_are_rejected(key):
    with pytest.raises(ValueError, match="identity"):
        SmallPredictor("ridge_delta").fit(
            [{key: "a"}, {key: "b"}], [0.2, 0.3], [0.1, 0.1], ["a", "b"]
        )


@pytest.mark.parametrize(
    "row",
    [{"x": float("inf")}, {"x": []}, {"x": {}}, {"x": object()}, {"": 1}, {1: 2}, {"x\x1fy": 1}],
)
def test_invalid_feature_shapes_and_values_rejected(row):
    with pytest.raises((ValueError, TypeError)):
        SmallPredictor("ridge_delta").fit([row], [0.2], [0.1], ["a"])


def test_mixed_training_or_changed_prediction_types_rejected():
    with pytest.raises(ValueError, match="mix"):
        SmallPredictor("ridge_delta").fit(
            [{"x": 1}, {"x": "one"}], [0.2, 0.3], [0.1, 0.1], ["a", "b"]
        )
    model = fitted()
    with pytest.raises(ValueError, match="type"):
        model.predict([{"lr": "not numeric"}], [0.2])


@pytest.mark.parametrize("groups", [[], ["a"], ["a", None], ["a", []], ["a", True], ["a", ""]])
def test_bad_group_vectors_rejected(groups):
    with pytest.raises(ValueError):
        SmallPredictor("ridge_delta").fit([{"x": 1}, {"x": 2}], [0.2, 0.3], [0.1, 0.1], groups)


@pytest.mark.parametrize(
    "embedding",
    [None, [1, 2], [[1, 2]], np.empty((2, 0)), [[float("nan")], [1]], [[float("inf")], [1]]],
)
def test_bad_embedding_dimensions_or_values_rejected(embedding):
    with pytest.raises(ValueError):
        SmallPredictor("embedding_ridge_delta").fit(
            [{"x": 1}, {"x": 2}], [0.2, 0.3], [0.1, 0.1], ["a", "b"], embedding
        )


def test_embedding_fit_predict_contract_and_nonembedding_mismatch_rejected():
    with pytest.raises(ValueError, match="two"):
        SmallPredictor("embedding_ridge_delta").fit([{"x": 1}], [0.2], [0.1], ["a"], [[1, 2]])
    with pytest.raises(ValueError, match="Nonembedding"):
        SmallPredictor("ridge_delta").fit([{"x": 1}], [0.2], [0.1], ["a"], [[1, 2]])
    model = fitted("embedding_ridge_delta")
    for embeddings in (None, np.ones((1, 23)), np.ones((2, 24))):
        with pytest.raises(ValueError):
            model.predict([{"lr": 1e-5}], [0.2], embeddings)
    with pytest.raises(ValueError, match="Nonembedding"):
        fitted().predict([{"lr": 1e-5}], [0.2], np.ones((1, 24)))


def test_empty_batches_unfitted_prediction_and_reference_lengths():
    with pytest.raises(ValueError, match="training row"):
        SmallPredictor("ridge_delta").fit([], [], [], [])
    with pytest.raises(ValueError, match="training feature"):
        SmallPredictor("ridge_delta").fit([{}], [0.2], [0.1], ["a"])
    with pytest.raises(ValueError, match="fitted"):
        SmallPredictor("ridge_delta").predict([{"x": 1}], [0.1])
    model = fitted()
    assert model.predict([], []).shape == (0,)
    with pytest.raises(ValueError, match="length"):
        model.predict([{"lr": 1e-5}], [])


def test_failed_refit_preserves_previous_fitted_state():
    model = fitted()
    before = pickle.dumps(model)
    with pytest.raises(ValueError):
        model.fit([{"lr": 1e-5}], [2.0], [0.2], ["a"])
    assert pickle.dumps(model) == before


def test_relabeling_groups_without_changing_membership_does_not_change_predictions():
    rows, y, reference, groups, _ = fixture()
    first = SmallPredictor("ridge_delta").fit(rows, y, reference, groups)
    second = SmallPredictor("ridge_delta").fit(
        rows, y, reference, ["renamed-" + group for group in groups]
    )
    np.testing.assert_array_equal(first.predict(rows, reference), second.predict(rows, reference))
    assert not any("session-" in name for name in first.vectorizer_.vocabulary_)
