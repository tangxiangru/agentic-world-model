"""Fixed lightweight predictors for the strictly screened recorder refresh.

No feature extraction, model selection, or dataset access happens here. The
caller supplies positive numeric features and an explicit reference vector.
An unknown reference encoded as zero is an offset convention, NOT an observed
base/parent accuracy. Except for ``old_absolute``, the learned target is the
child accuracy minus that caller-supplied reference.

All arms preserve the old table's unnormalized sample weight 1/n_per_session.
Each session therefore contributes total weight one. This matters for L2/ridge
regularization: normalizing weights to mean one is not equivalent at fixed L2.

Old-style HGB: 300 rounds, .04 learning rate, eight leaves, leaf minimum five,
L2 one, no depth cap, average seeds 0/1/2. Early stopping is explicitly disabled
(equivalent to the old auto setting at the present <10,000-example scale).
Its vocabulary is train-only; NaNs stay native except all-training-NaN columns
are set to zero in BOTH train and test. No missingness indicators are added.

Only configuration declarations supplied by the caller are modeled. These
classes do not revive old unsafe feature extraction or certify parent identity.
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
from numbers import Real

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

NAMES = ("old_delta", "old_absolute", "regularized_delta", "ridge_delta")
OLD_HGB_PARAMETERS = {
    "loss": "squared_error",
    "max_iter": 300,
    "learning_rate": 0.04,
    "max_leaf_nodes": 8,
    "min_samples_leaf": 5,
    "l2_regularization": 1.0,
    "max_depth": None,
    "early_stopping": False,
}
REGULARIZED_HGB_PARAMETERS = {
    "loss": "squared_error",
    "max_iter": 200,
    "learning_rate": 0.04,
    "max_leaf_nodes": 4,
    "min_samples_leaf": 10,
    "l2_regularization": 10.0,
    "max_depth": 2,
    "early_stopping": False,
}
RIDGE_ALPHA = 100.0
IDENTITY_COMPONENT = re.compile(
    r"(?:^|[./:\s_-])(?:id|ids|example_id|cell_id|run_id|session_id|group_id|"
    r"candidate_id|parent_id|checkpoint_id|recipe_id)(?:$|[./:\s_-])|"
    r"(?:^|[./:\s-])(?:groups?|cell|session|run)(?:$|[./:\s-])"
)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _features(features):
    if isinstance(features, (str, bytes, Mapping)):
        raise TypeError("Features must be a sequence of numeric dictionaries")
    try:
        supplied = list(features)
    except TypeError as error:
        raise TypeError("Features must be a sequence of numeric dictionaries") from error
    result = []
    for row in supplied:
        if not isinstance(row, Mapping):
            raise TypeError("Every feature row must be a numeric dictionary")
        projected = {}
        for key, value in row.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Feature keys must be nonempty strings")
            if IDENTITY_COMPONENT.search(key):
                raise ValueError("Record/session/group identity must not be predictive")
            if value is None:
                value = np.nan
            elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError("Features must be numeric or None, not booleans or text")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Feature exceeds supported floating-point range") from error
            if math.isinf(numeric):
                raise ValueError("Infinite features are forbidden")
            projected[key] = numeric
        result.append(projected)
    return result


def _accuracy(values, expected, name):
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(name + " must be a numeric vector")
    try:
        supplied = list(values)
    except TypeError as error:
        raise TypeError(name + " must be a numeric vector") from error
    if len(supplied) != expected:
        raise ValueError(name + " length must match features")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in supplied
    ):
        raise TypeError(name + " must contain numbers, not booleans or text")
    try:
        result = np.asarray(supplied, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name + " exceeds supported floating-point range") from error
    if result.ndim != 1 or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError(name + " must be a finite one-dimensional vector in [0,1]")
    return result


def _weights(groups, expected):
    if isinstance(groups, (str, bytes, Mapping)):
        raise TypeError("Groups must be a sequence of session identifiers")
    try:
        supplied = list(groups)
    except TypeError as error:
        raise TypeError("Groups must be a sequence of session identifiers") from error
    if len(supplied) != expected:
        raise ValueError("Groups length must match features")
    if any(not isinstance(group, str) or not group.strip() for group in supplied):
        raise TypeError("Groups must be nonempty string session identifiers")
    counts = Counter(supplied)
    return np.asarray([1.0 / counts[group] for group in supplied]), counts


def _matrix(rows, keys, vectorizer, *, fit=False):
    # DictVectorizer normally treats absent keys as numeric zero. Here omission
    # means unknown, consistently with an explicit None/NaN for a trained key.
    projected = [{key: row.get(key, np.nan) for key in keys} for row in rows]
    return vectorizer.fit_transform(projected) if fit else vectorizer.transform(projected)


class CleanPredictor:
    """A frozen model family; feature views and evaluation splits are external."""

    def __init__(self, name):
        if name not in NAMES:
            raise ValueError("Unknown predictor; expected one of " + ", ".join(NAMES))
        self.name = name
        self._fitted = False

    def fit(self, features, labels, references, groups):
        started = time.perf_counter()
        rows = _features(features)
        if not rows:
            raise ValueError("Training features must not be empty")
        y = _accuracy(labels, len(rows), "Labels")
        reference = _accuracy(references, len(rows), "References")
        weights, counts = _weights(groups, len(rows))
        keys = sorted({key for row in rows for key in row})
        if not keys:
            raise ValueError("At least one positive feature key is required")
        vectorizer = DictVectorizer(sparse=False, dtype=np.float64)
        raw = _matrix(rows, keys, vectorizer, fit=True)
        all_missing = np.isnan(raw).all(axis=0)
        imputer = scaler = None
        matrix = raw.copy()
        if self.name == "ridge_delta":
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            matrix = np.concatenate(
                (imputer.fit_transform(raw), np.isnan(raw).astype(float)), axis=1
            )
            scaler = StandardScaler()
            matrix = scaler.fit_transform(matrix, sample_weight=weights)
            if not np.isfinite(matrix).all():
                raise ValueError("Ridge preprocessing produced nonfinite values")
            estimators = [Ridge(alpha=RIDGE_ALPHA, solver="svd")]
            parameters = None
            seeds = []
        else:
            matrix[:, all_missing] = 0.0
            parameters = copy.deepcopy(
                REGULARIZED_HGB_PARAMETERS
                if self.name == "regularized_delta"
                else OLD_HGB_PARAMETERS
            )
            seeds = [0] if self.name == "regularized_delta" else [0, 1, 2]
            estimators = [
                HistGradientBoostingRegressor(**parameters, random_state=seed) for seed in seeds
            ]
        target = y if self.name == "old_absolute" else y - reference
        preprocessed = time.perf_counter()
        with threadpool_limits(limits=1):
            for estimator in estimators:
                estimator.fit(matrix, target, sample_weight=weights)
        finished = time.perf_counter()
        # Publish state only after all validation and fitting have succeeded.
        self.keys_ = keys
        self.vectorizer_ = vectorizer
        self.all_missing_ = all_missing
        self.imputer_ = imputer
        self.scaler_ = scaler
        self.estimators_ = estimators
        self._metadata = {
            "schema": "wm-clean-predictor-v1",
            "name": self.name,
            "target": "child_accuracy"
            if self.name == "old_absolute"
            else "child_minus_caller_reference",
            "reference_is_certified_parent_accuracy": False,
            "unknown_reference_convention": "Caller may supply zero offset; zero is not asserted base accuracy",
            "train_rows": len(rows),
            "train_groups": len(counts),
            "raw_feature_width": raw.shape[1],
            "model_feature_width": matrix.shape[1],
            "all_training_missing_columns": int(all_missing.sum()),
            "missing_policy": "train_median_plus_every_field_indicator_then_weighted_scaling"
            if self.name == "ridge_delta"
            else "native_nan_except_all_training_missing_columns_zeroed",
            "feature_vocabulary_sha256": _digest(vectorizer.vocabulary_),
            "session_counts_sha256": _digest(dict(counts)),
            "weight_policy": "unnormalized_1_over_number_of_training_cards_in_session",
            "weight_sum": float(weights.sum()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
            "group_ids_used_as_features": False,
            "hgb_parameters": parameters,
            "hgb_seeds": seeds,
            "ensemble_members": len(estimators),
            "ridge_parameters": {"alpha": RIDGE_ALPHA, "solver": "svd"}
            if self.name == "ridge_delta"
            else None,
            "preprocessing_seconds": preprocessed - started,
            "estimator_fit_seconds": finished - preprocessed,
            "fit_seconds": finished - started,
            "output_clip": [0.0, 1.0],
        }
        self._fitted = True
        return self

    def predict(self, features, references):
        if not self._fitted:
            raise ValueError("Predictor has not been fitted")
        rows = _features(features)
        reference = _accuracy(references, len(rows), "References")
        if not rows:
            return np.empty(0, dtype=np.float64)
        raw = _matrix(rows, self.keys_, self.vectorizer_)
        if self.name == "ridge_delta":
            matrix = np.concatenate(
                (self.imputer_.transform(raw), np.isnan(raw).astype(float)), axis=1
            )
            matrix = self.scaler_.transform(matrix)
            if not np.isfinite(matrix).all():
                raise ValueError("Ridge preprocessing produced nonfinite values")
        else:
            matrix = raw.copy()
            matrix[:, self.all_missing_] = 0.0
        with threadpool_limits(limits=1):
            predictions = np.mean(
                [estimator.predict(matrix) for estimator in self.estimators_], axis=0
            )
        if self.name != "old_absolute":
            predictions = predictions + reference
        if not np.isfinite(predictions).all():
            raise ValueError("Model produced nonfinite predictions")
        return np.clip(predictions, 0.0, 1.0)

    def metadata(self):
        if not self._fitted:
            return {"name": self.name, "fitted": False}
        return {**copy.deepcopy(self._metadata), "fitted": True}
