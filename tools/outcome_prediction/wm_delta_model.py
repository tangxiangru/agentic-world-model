"""Small parent-conditioned baselines; callers own splits and target construction.

Only the explicitly named row fields below are read. Child labels enter fit via
the separate targets argument, never through a row. History is oldest to newest;
its known scores are permitted inputs under this interface. No grouping, search,
data loading, model calls, or evaluation is performed here.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from numbers import Real

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SEED = 20260905
NAMES = frozenset(
    {
        "carry_parent",
        "ridge_current_delta",
        "ridge_history_delta",
        "trees_history_delta",
        "ridge_history_absolute",
    }
)
DECAY = 0.5


def _number(value, name, *, accuracy=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value) or (accuracy and not 0 <= value <= 1):
        raise ValueError(f"Invalid {name}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _numeric_features(value):
    if not isinstance(value, dict):
        raise TypeError("features must be a dictionary")
    return {
        _text(key, "feature key"): _number(number, "feature value") for key, number in value.items()
    }


def _rows(rows, *, benchmark=None, allow_empty=False):
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Each row must be a dictionary")
        try:
            _text(row["example_id"], "example_id")
            _text(row["parent_key"], "parent_key")
            cell = _text(row["cell_id"], "cell_id")
            task = _text(row["benchmark"], "benchmark")
            parent = _number(row["parent_accuracy"], "parent accuracy", accuracy=True)
            current = _numeric_features(row["current_features"])
            raw_history = row["history"]
            if not isinstance(raw_history, list):
                raise TypeError("history must be an oldest-to-newest list")
            history = []
            for entry in raw_history:
                if not isinstance(entry, dict):
                    raise TypeError("History entries must be dictionaries")
                score = entry["accuracy"]
                history.append(
                    {
                        "features": _numeric_features(entry["features"]),
                        "accuracy": None
                        if score is None
                        else _number(score, "history accuracy", accuracy=True),
                        "relation": _text(entry["relation"], "history relation"),
                    }
                )
        except KeyError as exc:
            raise ValueError("Missing required delta-model input field") from exc
        if benchmark is None:
            benchmark = task
        if task != benchmark:
            raise ValueError("Benchmark mixing is not supported")
        result.append(
            {"cell_id": cell, "parent_accuracy": parent, "current": current, "history": history}
        )
    if not result and not allow_empty:
        raise ValueError("At least one training row is required")
    return result, benchmark


def _column(kind, key):
    # Delimit/escape user feature names so prefixes and punctuation cannot collide.
    return kind + ":" + json.dumps(key, ensure_ascii=False)


def _features(row, *, history):
    out = {"parent_accuracy": row["parent_accuracy"]}
    for key, value in row["current"].items():
        out[_column("current_value", key)] = value
        out[_column("current_present", key)] = 1.0
    if history:
        entries = row["history"]
        known = [entry["accuracy"] for entry in entries if entry["accuracy"] is not None]
        out.update(
            {
                "history_count": float(len(entries)),
                "history_known_accuracy_count": float(len(known)),
                "history_accuracy_present": float(bool(known)),
                "history_accuracy_mean": math.fsum(known) / len(known) if known else 0.0,
                "history_accuracy_recent": known[-1] if known else 0.0,
                "history_last_gain_present": float(len(known) >= 2),
                "history_last_gain": known[-1] - known[-2] if len(known) >= 2 else 0.0,
            }
        )
        values = defaultdict(list)
        for position, entry in enumerate(entries):
            relation = _column("history_relation_count", entry["relation"])
            out[relation] = out.get(relation, 0.0) + 1.0
            for key, value in entry["features"].items():
                values[key].append((position, value))
        for key, observed in values.items():
            last_position, recent = observed[-1]
            # Relative to last presence: identical normalized weights but no
            # all-zero underflow when the feature was last seen long ago.
            weighted = [(DECAY ** (last_position - i), v) for i, v in observed]
            try:
                total = math.fsum(v for _, v in observed)
                mean = total / len(observed)
                recency_mean = math.fsum(w * v for w, v in weighted) / math.fsum(
                    w for w, _ in weighted
                )
            except OverflowError as exc:
                raise ValueError("Nonfinite derived history feature") from exc
            for kind, value in (
                ("history_feature_count", float(len(observed))),
                ("history_sum", total),
                ("history_mean", mean),
                ("history_recent", recent),
                ("history_recency_mean", recency_mean),
            ):
                out[_column(kind, key)] = value
            if key in row["current"]:
                current = row["current"][key]
                out[_column("current_minus_history_mean", key)] = current - mean
                out[_column("current_minus_history_recent", key)] = current - recent
                out[_column("current_times_history_recent", key)] = current * recent
    if not all(math.isfinite(value) for value in out.values()):
        raise ValueError("Nonfinite derived feature")
    return out


class LightweightDeltaModel:
    """Fixed small estimator with signed raw deltas and clipped accuracy outputs."""

    def __init__(self, name):
        if name not in NAMES:
            raise ValueError("Unknown lightweight delta model")
        self.name = name
        self._fitted = False

    def fit(self, rows, targets):
        checked, benchmark = _rows(rows)
        y = np.asarray([_number(y, "child target", accuracy=True) for y in targets])
        if len(y) != len(checked):
            raise ValueError("Rows and targets must have identical lengths")
        counts = Counter(row["cell_id"] for row in checked)
        weights = np.asarray([1.0 / counts[row["cell_id"]] for row in checked])
        weights /= weights.mean()
        parent = np.asarray([row["parent_accuracy"] for row in checked])
        use_history = "history" in self.name
        vectorizer, scaler, estimator = None, None, None
        if self.name != "carry_parent":
            vectorizer = DictVectorizer(sparse=False, sort=True)
            x = vectorizer.fit_transform([_features(row, history=use_history) for row in checked])
            if self.name.startswith("ridge_"):
                scaler = StandardScaler()
                x = scaler.fit_transform(x, sample_weight=weights)
                estimator = Ridge(alpha=10.0, solver="lsqr", tol=1e-4)
            else:
                estimator = ExtraTreesRegressor(
                    n_estimators=64,
                    max_depth=3,
                    min_samples_leaf=3,
                    n_jobs=1,
                    random_state=SEED,
                )
            estimator.fit(
                x,
                y if self.name == "ridge_history_absolute" else y - parent,
                sample_weight=weights,
            )
        self.benchmark_ = benchmark
        self.vectorizer_, self.scaler_, self.estimator_ = vectorizer, scaler, estimator
        self.sample_weight_ = weights
        self.feature_names_ = (
            [] if vectorizer is None else vectorizer.get_feature_names_out().tolist()
        )
        self.n_rows_, self.n_cells_ = len(checked), len(counts)
        self._fitted = True
        return self

    def predict_delta(self, rows):
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
        checked, _ = _rows(rows, benchmark=self.benchmark_, allow_empty=True)
        if not checked or self.name == "carry_parent":
            return np.zeros(len(checked), dtype=float)
        x = self.vectorizer_.transform(
            [_features(row, history="history" in self.name) for row in checked]
        )
        assert x.shape[1] == len(self.feature_names_), "History must not expand fitted width"
        if self.scaler_ is not None:
            x = self.scaler_.transform(x)
        prediction = np.asarray(self.estimator_.predict(x), dtype=float)
        if self.name == "ridge_history_absolute":
            prediction -= np.asarray([row["parent_accuracy"] for row in checked])
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite model prediction")
        return prediction

    def predict_accuracy(self, rows):
        rows = list(rows)
        delta = self.predict_delta(rows)
        parents = np.asarray([_number(r["parent_accuracy"], "parent accuracy") for r in rows])
        return np.clip(parents + delta, 0.0, 1.0)

    def metadata(self):
        if not self._fitted:
            raise RuntimeError("fit must be called before metadata")
        return {
            "schema": "lightweight-delta-model-v1",
            "name": self.name,
            "benchmark": self.benchmark_,
            "training_rows": self.n_rows_,
            "training_cells": self.n_cells_,
            "feature_count": len(self.feature_names_),
            "feature_names": list(self.feature_names_),
            "target": "child_accuracy" if self.name.endswith("absolute") else "child_minus_parent",
            "raw_delta_is_signed": True,
            "accuracy_clipping": [0.0, 1.0],
            "sample_weight": "equal total per cell; arithmetic mean one",
            "history_order": "oldest_to_newest; recent accuracy skips unknown scores",
            "history_last_gain": "last known accuracy minus previous known accuracy",
            "history_recency_decay": DECAY,
            "step_index_feature_columns": False,
            "preprocessing": "train-only DictVectorizer; train-weighted StandardScaler for ridge",
            "estimator_parameters": None
            if self.estimator_ is None
            else self.estimator_.get_params(deep=False),
            "identity_fields_predictive": False,
            "classification_or_pair_grouping": "caller responsibility; equal scores do not identify parents",
        }
