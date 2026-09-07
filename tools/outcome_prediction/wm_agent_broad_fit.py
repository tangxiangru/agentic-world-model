"""All-training-evidence lightweight fits with bounded agent relevance weights.

Every clean same-benchmark training example enters each query fit. The agent
chooses bounded weights, up to twelve existing or audited data features, and an
accuracy/delta target plus a fixed model. A 50% session-balanced uniform component
limits concentration. Missing parent/target scores remain invalid for both target
choices. No data loading, LLM calls, query outcomes, or fitted-prediction overrides.

Leave-one-session-out diagnostics are conditional on an agent policy that may
have seen those training outcomes, and are not unbiased policy validation.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from tools.outcome_prediction import wm_agent_local_fit as old
from tools.outcome_prediction.wm_agent_recipe_data import FEATURE_KEYS as DATA_FEATURE_KEYS

FEATURE_KEYS = tuple(sorted((*old.FEATURE_KEYS, *DATA_FEATURE_KEYS)))
MODELS = old.MODELS
TARGETS = ("accuracy", "delta")
SEED = old.SEED
MAX_FEATURES = 12
EXPECTED_TRAIN_COUNTS = {"gsm8k": 51, "aime2025": 72}
SPEC_KEYS = frozenset({"weights", "target", "model", "feature_keys", "rationale"})
WEIGHT_POLICY = (
    "All same-benchmark training rows participate. Agent relevance in [0.5,2] is divided by "
    "full training session count and normalized to mean one; mix 50% of this with 50% "
    "exactly session-balanced uniform weights, also mean one. No subset filtering."
)


def validate_row(row, *, training=False):
    """Exact augmented history schema; frozen validator checks original151 inputs."""
    if not isinstance(row, Mapping) or not isinstance(row.get("views"), Mapping):
        raise ValueError("Expected an input-only row with feature views")  # noqa: TRY004
    history = row["views"].get("history")
    if not isinstance(history, Mapping) or set(history) != set(FEATURE_KEYS):
        raise ValueError("History must contain exactly the frozen augmented feature schema")
    for name, view in row["views"].items():
        if not isinstance(view, Mapping) or set(view) - set(FEATURE_KEYS):
            raise ValueError("Only whitelisted numeric feature views are permitted")
        if name != "history" and set(view) - set(old.FEATURE_KEYS):
            raise ValueError("Augmented data features belong only in the history view")
        for key, value in view.items():
            old._finite(value, "Feature " + key)
    projected = {**row, "views": {name: {key: value for key, value in view.items()
                                          if key in old.FEATURE_KEYS}
                                   for name, view in row["views"].items()}}
    old._validate_row(projected, training=training)


def validate_spec(spec, all_train, query):
    """Validate full training evidence and bounded model specification, no labels."""
    validate_row(query)
    if not isinstance(spec, Mapping) or set(spec) != SPEC_KEYS:
        raise ValueError("Use exactly weights, target, model, feature_keys, rationale; no subset selection")
    if not isinstance(all_train, (list, tuple)) or len(all_train) < 8:
        raise ValueError("At least eight full-pool training examples are required")
    expected = EXPECTED_TRAIN_COUNTS.get(query["benchmark"])
    if expected is not None and len(all_train) != expected:
        raise ValueError("All frozen benchmark training examples are required; subset selection forbidden")
    ids = set()
    for row in all_train:
        validate_row(row, training=True)
        if row["example_id"] in ids:
            raise ValueError("Duplicate training identity")
        ids.add(row["example_id"])
        if row["benchmark"] != query["benchmark"]:
            raise ValueError("Every training example must match the query benchmark")
        if row["cell_id"] == query["cell_id"] or row["example_id"] == query["example_id"]:
            raise ValueError("Training/query identity or session overlap")
    if len({row["cell_id"] for row in all_train}) < 4:
        raise ValueError("At least four independent training sessions are required")
    weights = spec["weights"]
    if not isinstance(weights, list) or len(weights) != len(all_train):
        raise ValueError("One relevance weight is required per full training row, in row order")
    weights = [old._finite(weight, "Relevance weight") for weight in weights]
    if any(not 0.5 <= weight <= 2 for weight in weights):
        raise ValueError("Relevance weights must be in [0.5,2]")
    if not isinstance(spec["target"], str) or spec["target"] not in TARGETS:
        raise ValueError("Target must be accuracy or delta")
    if not isinstance(spec["model"], str) or spec["model"] not in MODELS:
        raise ValueError("Unknown lightweight model")
    keys = spec["feature_keys"]
    if (not isinstance(keys, list) or len(keys) > MAX_FEATURES
            or any(not isinstance(key, str) for key in keys)
            or len(keys) != len(set(keys)) or set(keys) - set(FEATURE_KEYS)):
        raise ValueError("Choose at most twelve distinct exact whitelisted feature keys")
    if spec["model"] != "weighted_median" and "parent.accuracy" not in keys:
        raise ValueError("Learned models require parent.accuracy")
    if not isinstance(spec["rationale"], str) or not spec["rationale"].strip():
        raise ValueError("A nonempty rationale is required")
    return {"weights": weights, "target": spec["target"], "model": spec["model"],
            "feature_keys": list(keys), "rationale": spec["rationale"]}


def effective_weights(rows, relevance):
    """Convex mixture of adjusted relevance and exactly session-balanced uniform."""
    counts = Counter(row["cell_id"] for row in rows)
    uniform = np.asarray([1.0 / counts[row["cell_id"]] for row in rows])
    uniform *= len(rows) / uniform.sum()
    adjusted = np.asarray([weight / counts[row["cell_id"]]
                           for row, weight in zip(rows, relevance)])
    adjusted *= len(rows) / adjusted.sum()
    mixed = 0.5 * uniform + 0.5 * adjusted
    if len(mixed) != len(rows) or not np.isfinite(mixed).all() or (mixed <= 0).any():
        raise ValueError("Invalid broad evidence weights")
    return mixed, uniform, adjusted


def _weighted_scaling(x, weights):
    """Stable weighted standardization, excluding constants before variance math.

    Work in bounded coordinates relative to the first training value and column
    spread. This avoids negative roundoff variances for constant tiny learning
    rates and reduces cancellation/overflow. Query values never set statistics.
    """
    anchor = x[0].copy()
    unit = np.ones(x.shape[1])
    center = np.zeros(x.shape[1])
    scale = np.ones(x.shape[1])
    active = np.zeros(x.shape[1], dtype=bool)
    transformed = np.zeros_like(x)
    normalized = weights / weights.sum()
    for column in range(x.shape[1]):
        differences = np.asarray([old._finite(float(value) - float(anchor[column]), "Feature spread")
                                  for value in x[:, column]])
        spread = float(np.max(np.abs(differences)))
        if spread == 0:
            continue
        if not math.isfinite(spread):
            raise ValueError("Feature numerical range exceeds safe scaling")
        values = differences / spread
        mean = math.fsum(float(w * value) for w, value in zip(normalized, values))
        variance = math.fsum(float(w) * (float(value) - mean) ** 2
                             for w, value in zip(normalized, values))
        if variance <= 0:
            continue
        unit[column], center[column], scale[column], active[column] = spread, mean, math.sqrt(variance), True
        transformed[:, column] = (values - mean) / scale[column]
    return transformed, {"anchor": anchor.tolist(), "unit": unit.tolist(),
                         "center": center.tolist(), "scale": scale.tolist(), "active": active.tolist()}


def _fit_state(rows, labels, spec, weights):
    label_key = "accuracy" if spec["target"] == "accuracy" else "delta_accuracy"
    y = np.asarray([labels[row["example_id"]][label_key] for row in rows])
    state = {"schema": "agent-broad-fit-state-v1", "model": spec["model"],
             "target": spec["target"], "feature_keys": list(spec["feature_keys"])}
    if spec["model"] == "weighted_median":
        order = np.argsort(y, kind="stable")
        index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left")
        state["constant_value"] = float(y[order[min(int(index), len(order) - 1)]])
        return state
    x = np.asarray([[row["views"]["history"][key] for key in spec["feature_keys"]] for row in rows])
    if spec["model"].startswith("ridge_"):
        transformed, scaler = _weighted_scaling(x, weights)
        alpha = int(spec["model"].split("_")[1])
        model = Ridge(alpha=alpha, solver="svd").fit(transformed, y, sample_weight=weights)
        state.update(alpha=alpha, scaler=scaler, coefficients=model.coef_.tolist(), intercept=float(model.intercept_))
    else:
        model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, random_state=SEED)
        model.fit(x, y, sample_weight=weights)
        tree = model.tree_
        state.update({"max_depth": 2, "min_samples_leaf": 5, "seed": SEED,
                      "children_left": tree.children_left.tolist(), "children_right": tree.children_right.tolist(),
                      "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                      "value": tree.value[:, 0, 0].tolist(), "node_samples": tree.n_node_samples.tolist()})
    return state


def predict_from_state(state, query):
    """Replay a frozen numerical forecast with no target labels or LLM override."""
    validate_row(query)
    if (state.get("schema") != "agent-broad-fit-state-v1" or state.get("model") not in MODELS
            or state.get("target") not in TARGETS):
        raise ValueError("Invalid serialized broad-evidence model")
    if state["model"] == "weighted_median":
        raw = old._finite(state["constant_value"], "Fitted constant")
    else:
        keys = state["feature_keys"]
        if not keys or len(keys) > MAX_FEATURES or set(keys) - set(FEATURE_KEYS):
            raise ValueError("Serialized model contains non-whitelisted features")
        x = np.asarray([query["views"]["history"][key] for key in keys])
        if state["model"].startswith("ridge_"):
            scaler = state["scaler"]
            active = np.asarray(scaler["active"], dtype=bool)
            transformed = np.zeros_like(x)
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    transformed[active] = (((x[active] - np.asarray(scaler["anchor"])[active])
                                            / np.asarray(scaler["unit"])[active])
                                           - np.asarray(scaler["center"])[active]) / np.asarray(scaler["scale"])[active]
                    raw = float(transformed @ np.asarray(state["coefficients"]) + state["intercept"])
            except FloatingPointError as exc:
                raise ValueError("Query exceeds safe standardized numerical range") from exc
        else:
            x = x.astype(np.float32)
            node = 0
            while state["children_left"][node] != -1:
                direction = "children_left" if x[state["feature"][node]] <= state["threshold"][node] else "children_right"
                node = state[direction][node]
            raw = float(state["value"][node])
    raw = old._finite(raw, "Fitted prediction")
    reference = old._accuracy(query["parent_reference"])
    unclipped = raw if state["target"] == "accuracy" else reference + raw
    predicted = float(np.clip(unclipped, 0, 1))
    return {"predicted_accuracy": predicted, "predicted_delta": predicted - reference,
            "raw_prediction": raw, "raw_predicted_delta": unclipped - reference,
            "unclipped_predicted_accuracy": unclipped, "clipped": predicted != unclipped}


def _session_mae(rows, labels, predictions):
    groups = sorted({row["cell_id"] for row in rows})
    errors = {group: [] for group in groups}
    for row, prediction in zip(rows, predictions):
        errors[row["cell_id"]].append(abs(prediction - labels[row["example_id"]]["accuracy"]))
    return math.fsum(math.fsum(values) / len(values) for values in errors.values()) / len(groups)


def fit_predict(all_train, train_labels, query, spec):
    """Fit every clean training row; query outcomes are not accepted anywhere."""
    spec = validate_spec(spec, all_train, query)
    ids = [row["example_id"] for row in all_train]
    if not isinstance(train_labels, Mapping) or set(train_labels) != set(ids):
        raise ValueError("Labels must match exactly all training rows, with no held-out labels")
    for row in all_train:
        label = train_labels[row["example_id"]]
        if not isinstance(label, Mapping):
            raise ValueError("A valid official training label is required")  # noqa: TRY004
        target = old._accuracy(label.get("accuracy"))
        delta = old._finite(label.get("delta_accuracy"), "Delta label")
        if delta != target - row["parent_reference"]:
            raise ValueError("Delta must equal valid target minus valid parent; no accuracy imputation")
    weights, uniform, adjusted = effective_weights(all_train, spec["weights"])
    state = _fit_state(all_train, train_labels, spec, weights)
    forecast = predict_from_state(state, query)
    groups = sorted({row["cell_id"] for row in all_train})
    session_totals = [math.fsum(float(weights[i]) for i, row in enumerate(all_train) if row["cell_id"] == group)
                      for group in groups]
    fitted = [predict_from_state(state, row)["predicted_accuracy"] for row in all_train]
    fold_records = []
    for group in groups:
        indices = [i for i, row in enumerate(all_train) if row["cell_id"] != group]
        rows = [all_train[i] for i in indices]
        validation = [row for row in all_train if row["cell_id"] == group]
        labels = {row["example_id"]: train_labels[row["example_id"]] for row in rows}
        fold_weights, _, _ = effective_weights(rows, [spec["weights"][i] for i in indices])
        fold_state = _fit_state(rows, labels, spec, fold_weights)
        predictions = [predict_from_state(fold_state, row)["predicted_accuracy"] for row in validation]
        fold_records.append({"held_out_training_session": group, "fit_ids": [row["example_id"] for row in rows],
                             "validation_ids": [row["example_id"] for row in validation],
                             "mae": _session_mae(validation, train_labels, predictions)})
    return {
        **forecast, "spec": spec, "model_state": state,
        "fit_metadata": {
            "training_ids": ids, "training_examples": len(ids), "training_sessions": len(groups),
            "normalized_sample_weights": weights.tolist(), "session_balanced_component": uniform.tolist(),
            "normalized_relevance_component": adjusted.tolist(), "session_totals": dict(zip(groups, session_totals)),
            "row_effective_sample_size": float(weights.sum() ** 2 / (weights @ weights)),
            "session_effective_sample_size": math.fsum(session_totals) ** 2 / math.fsum(value ** 2 for value in session_totals),
            "weight_policy": WEIGHT_POLICY, "target": spec["target"],
            "feature_width": 0 if spec["model"] == "weighted_median" else len(spec["feature_keys"]),
            "training_equal_session_mae": _session_mae(all_train, train_labels, fitted),
            "scaling": "Train-only weighted standardization in bounded relative coordinates; constant columns inactive before variance computation.",
        },
        "conditional_training_diagnostics": {
            "description": "Leave-one-session-out MAE conditional on the already agent-chosen weights/features/model/target; descriptive only, not unbiased policy validation.",
            "equal_session_mae": math.fsum(fold["mae"] for fold in fold_records) / len(fold_records),
            "folds": fold_records,
        },
    }
