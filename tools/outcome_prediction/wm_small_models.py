"""Small, CPU-only outcome regressors with strictly train-fitted preprocessing.

The caller supplies positively selected feature dictionaries and, optionally,
already computed embeddings in the same row order. This module loads no data,
embedding service, credentials, or experiment artifacts. Groups affect sample
weights only, never predictive features. Parent references are caller-provided
numbers, not a claim that an observed base-model score exists.

Missing numeric settings (absent, None, or NaN) are median-imputed, with an
explicit missing indicator for EVERY training feature. Unseen test features are
ignored; no vocabulary, medians, scales, or PCA components are fit at prediction.
Serialization may use trusted local pickle/joblib; never load untrusted pickle.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

NAMES = (
    "ridge_delta",
    "ridge_absolute",
    "hgb_delta",
    "hgb_absolute",
    "embedding_ridge_delta",
    "embedding_kernel_delta",
)
HGB_PARAMETERS = {
    "max_iter": 200,
    "max_leaf_nodes": 8,
    "min_samples_leaf": 5,
    "l2_regularization": 1.0,
    "learning_rate": 0.04,
    "max_depth": 3,
    "early_stopping": False,
}
_IDENTITY_COMPONENT = re.compile(
    r"(?:^|[./:\s_-])(?:id|ids|example_id|cell_id|run_id|session_id|group_id|"
    r"candidate_id|parent_id|checkpoint_id|recipe_id)(?:$|[./:\s_-])|"
    r"(?:^|[./:\s-])(?:groups?|cell|session|run)(?:$|[./:\s-])"
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _array_digest(value):
    array = np.asarray(value, dtype="<f8", order="C")
    return _digest(
        {"shape": list(array.shape), "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    )


def _accuracy_vector(values, n, name):
    try:
        items = list(values)
        if any(isinstance(v, (bool, np.bool_)) for v in items):
            raise ValueError(f"{name} must contain numeric accuracies, not booleans")
        array = np.asarray(items, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional numeric vector") from exc
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} length must match feature rows")
    if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError(f"{name} must contain finite accuracies in [0, 1]")
    return array


def _feature_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (Real, np.bool_)):
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("Numeric feature exceeds finite floating-point range") from exc
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ValueError("Infinite numeric features are not permitted")
        return number
    raise TypeError("Feature values must be numeric, categorical strings, or missing")


def _feature_rows(feature_dicts):
    rows = []
    for row in feature_dicts:
        if not isinstance(row, Mapping):
            raise TypeError("Each feature row must be a dictionary-like mapping")
        normalized = {}
        for key, value in row.items():
            if not isinstance(key, str) or not key.strip() or "\x1f" in key:
                raise ValueError(
                    "Feature names must be nonempty strings without reserved separators"
                )
            if _IDENTITY_COMPONENT.search(key.lower()):
                raise ValueError(
                    "Record/run/session/group identity must not be a predictive feature"
                )
            normalized[key] = _feature_value(value)
        rows.append(normalized)
    return rows


def _feature_types(rows):
    kinds = {}
    for row in rows:
        for key, value in row.items():
            kind = (
                None if value is None else ("categorical" if isinstance(value, str) else "numeric")
            )
            if key in kinds and kinds[key] is not None and kind not in (None, kinds[key]):
                raise ValueError("A feature cannot mix categorical and numeric values")
            if key not in kinds or kind is not None:
                kinds[key] = kind
    return {key: kinds[key] or "numeric" for key in sorted(kinds)}


def _encode_rows(rows, feature_types):
    encoded = []
    for row in rows:
        values = {}
        for key, kind in feature_types.items():
            value = row.get(key)
            if value is not None and (isinstance(value, str) != (kind == "categorical")):
                raise ValueError("Prediction feature type differs from its training type")
            values["missing\x1f" + key] = float(value is None)
            values["value\x1f" + key] = (
                ("missing" if value is None else "category:" + value)
                if kind == "categorical"
                else (np.nan if value is None else value)
            )
        encoded.append(values)
    return encoded


def _sample_weights(groups, n):
    values = list(groups)
    if len(values) != n or not values:
        raise ValueError("groups must match nonempty training feature rows")
    if any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (str, Integral))
        or (isinstance(v, str) and not v.strip())
        for v in values
    ):
        raise ValueError("Groups must be nonempty strings or integer identities")
    counts = Counter(values)
    weights = np.asarray([n / (len(counts) * counts[value]) for value in values], dtype=np.float64)
    return weights, len(counts)


def _embedding_array(embeddings, n, expected_width=None):
    if embeddings is None:
        raise ValueError("Embedding variants require externally supplied embeddings")
    try:
        array = np.asarray(embeddings, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Embeddings must be a finite two-dimensional numeric matrix") from exc
    if array.ndim != 2 or array.shape[0] != n or array.shape[1] < 1:
        raise ValueError("Embedding dimensions must be (feature rows, positive embedding width)")
    if expected_width is not None and array.shape[1] != expected_width:
        raise ValueError("Embedding width differs from training")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings must contain finite values")
    return array


class SmallPredictor:
    """One fixed small-data regressor; model selection and data splitting are external.

    ``alpha`` is passed through to ridge/kernel regularization (the runner can
    freeze a small grid such as 10 and 100). HGB always uses its fixed config.
    PCA is train-only and unweighted; estimator losses and StandardScaler use
    equal-total-per-group weights, normalized to mean one. No target or parent
    reference is implicitly appended to the features: the caller controls which
    known parent/history fields enter the positive feature dictionary.
    """

    def __init__(self, name, alpha=10, seed=20260905):
        if name not in NAMES:
            raise ValueError(f"Unknown small predictor name; expected one of {NAMES}")
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("alpha must be finite and positive")
        try:
            numeric_alpha = float(alpha)
        except OverflowError as exc:
            raise ValueError("alpha exceeds finite floating-point range") from exc
        if not math.isfinite(numeric_alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.name = name
        self.alpha = numeric_alpha
        self.seed = int(seed)
        self._fitted = False

    def fit(self, feature_dicts, child_accuracies, parent_reference, groups, embeddings=None):
        started = time.perf_counter()
        rows = _feature_rows(feature_dicts)
        if not rows:
            raise ValueError("At least one training row is required")
        y = _accuracy_vector(child_accuracies, len(rows), "child_accuracies")
        reference = _accuracy_vector(parent_reference, len(rows), "parent_reference")
        weights, group_count = _sample_weights(groups, len(rows))
        feature_types = _feature_types(rows)
        embedding_variant = self.name.startswith("embedding_")
        if not feature_types and not embedding_variant:
            raise ValueError("At least one training feature is required")
        if not embedding_variant and embeddings is not None:
            raise ValueError("Nonembedding variants must not receive embeddings")
        embedding = _embedding_array(embeddings, len(rows)) if embedding_variant else None
        if embedding_variant and len(rows) < 2:
            raise ValueError("Train-only PCA requires at least two training rows")

        vectorizer = DictVectorizer(sparse=False, dtype=np.float64, separator="\x1f")
        raw = vectorizer.fit_transform(_encode_rows(rows, feature_types))
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        matrix = imputer.fit_transform(raw) if raw.shape[1] else raw
        pca = None
        if embedding_variant:
            pca = PCA(n_components=min(16, len(rows) - 1, embedding.shape[1]), svd_solver="full")
            with threadpool_limits(limits=1):
                pca.fit(embedding)
                matrix = np.concatenate((matrix, pca.transform(embedding)), axis=1)
        scaler = None
        if not self.name.startswith("hgb_"):
            scaler = StandardScaler()
            matrix = scaler.fit_transform(matrix, sample_weight=weights)
        if not np.isfinite(matrix).all():
            raise ValueError("Preprocessing produced nonfinite values")
        gamma = None
        if self.name.startswith("hgb_"):
            estimator = HistGradientBoostingRegressor(**HGB_PARAMETERS, random_state=self.seed)
        elif self.name == "embedding_kernel_delta":
            gamma = 1.0 / matrix.shape[1]
            estimator = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=gamma)
        else:
            estimator = Ridge(alpha=self.alpha, solver="svd")
        target = y - reference if self.name.endswith("_delta") else y
        preprocessed = time.perf_counter()
        with threadpool_limits(limits=1):
            estimator.fit(matrix, target, sample_weight=weights)
        finished = time.perf_counter()

        # Publish the fitted state only after every validation/preprocessing/fit succeeds.
        self.feature_types_ = feature_types
        self.vectorizer_ = vectorizer
        self.imputer_ = imputer
        self.scaler_ = scaler
        self.pca_ = pca
        self.estimator_ = estimator
        self.embedding_width_ = embedding.shape[1] if embedding is not None else None
        self.structured_width_ = raw.shape[1]
        self._metadata = {
            "schema": "wm-small-predictor-v1",
            "name": self.name,
            "alpha": self.alpha,
            "alpha_used": not self.name.startswith("hgb_"),
            "seed": self.seed,
            "target": "child_minus_caller_reference"
            if self.name.endswith("_delta")
            else "child_accuracy",
            "parent_reference_is_asserted_observation": False,
            "train_rows": len(rows),
            "train_groups": group_count,
            "raw_feature_keys": len(feature_types),
            "structured_feature_width": raw.shape[1],
            "embedding_input_width": self.embedding_width_,
            "pca_components": int(pca.n_components_) if pca is not None else 0,
            "model_feature_width": matrix.shape[1],
            "sample_weight_policy": "equal_total_per_group_normalized_to_mean_one",
            "sample_weight_min": float(weights.min()),
            "sample_weight_max": float(weights.max()),
            "sample_weight_mean": float(weights.mean()),
            "weighted_standard_scaler": scaler is not None,
            "pca_weighting": "unweighted_train_rows" if pca is not None else None,
            "missing_indicators": "one_per_training_raw_feature_even_if_no_training_missingness",
            "vocabulary_sha256": _digest(vectorizer.vocabulary_),
            "feature_types_sha256": _digest(feature_types),
            "imputer_statistics_sha256": _array_digest(imputer.statistics_)
            if raw.shape[1]
            else None,
            "scaler_mean_sha256": _array_digest(scaler.mean_) if scaler is not None else None,
            "scaler_scale_sha256": _array_digest(scaler.scale_) if scaler is not None else None,
            "pca_components_sha256": _array_digest(pca.components_) if pca is not None else None,
            "pca_mean_sha256": _array_digest(pca.mean_) if pca is not None else None,
            "kernel_gamma": gamma,
            "hgb_parameters": copy.deepcopy(HGB_PARAMETERS)
            if self.name.startswith("hgb_")
            else None,
            "preprocessing_seconds": preprocessed - started,
            "estimator_fit_seconds": finished - preprocessed,
            "fit_seconds": finished - started,
            "group_identity_used_as_feature": False,
            "external_embedding_alignment": "caller_responsibility",
        }
        self._fitted = True
        return self

    def predict(self, feature_dicts, parent_reference, embeddings=None):
        if not self._fitted:
            raise ValueError("Predictor must be fitted before prediction")
        rows = _feature_rows(feature_dicts)
        reference = _accuracy_vector(parent_reference, len(rows), "parent_reference")
        embedding = None
        if self.embedding_width_ is not None:
            embedding = _embedding_array(embeddings, len(rows), self.embedding_width_)
        elif embeddings is not None:
            raise ValueError("Nonembedding variants must not receive embeddings")
        if not rows:
            return np.empty(0, dtype=np.float64)
        matrix = self.vectorizer_.transform(_encode_rows(rows, self.feature_types_))
        if self.structured_width_:
            matrix = self.imputer_.transform(matrix)
        with threadpool_limits(limits=1):
            if self.pca_ is not None:
                matrix = np.concatenate((matrix, self.pca_.transform(embedding)), axis=1)
            if self.scaler_ is not None:
                matrix = self.scaler_.transform(matrix)
            if not np.isfinite(matrix).all():
                raise ValueError("Prediction preprocessing produced nonfinite values")
            prediction = np.asarray(self.estimator_.predict(matrix), dtype=np.float64)
        if prediction.shape != (len(rows),) or not np.isfinite(prediction).all():
            raise ValueError("Estimator produced invalid predictions")
        if self.name.endswith("_delta"):
            prediction = prediction + reference
        return np.clip(prediction, 0.0, 1.0)

    def metadata(self):
        if not self._fitted:
            return {"name": self.name, "alpha": self.alpha, "seed": self.seed, "fitted": False}
        return {**copy.deepcopy(self._metadata), "fitted": True}
