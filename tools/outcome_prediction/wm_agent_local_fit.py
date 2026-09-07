"""Deterministic, label-isolated lightweight fits for an agent-selected subset.

The agent supplies a bounded data/model/feature selection. This numerical backend
never reads a dataset from disk, calls an LLM, selects a model using test labels,
or permits an LLM to override the fitted prediction. All target labels supplied
to ``fit_predict`` must belong to the selected training rows. Parent accuracy is
a required observed input, never imputed. Selection weights are divided by the
selected count in each session before global normalization, preserving relevance
weights without letting repeated cards automatically dominate a session.

Leave-one-session-out diagnostics are conditional on a subset and model already
chosen by the agent, potentially using all those training outcomes. They are
descriptive diagnostics, not unbiased validation of that selection procedure.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from tools.outcome_prediction.wm_one_step_features import (
    CURRENT_FIELDS,
    FAMILY_KEYS,
    HISTORY_FIELDS,
)

SEED = 20260906
MODELS = ("weighted_median", "ridge_10", "ridge_100", "shallow_tree")
MIN_EXAMPLES, MAX_EXAMPLES, MIN_SESSIONS, MAX_FEATURES = 8, 16, 4, 8
FEATURE_KEYS = tuple(sorted((
    "parent.accuracy", "parent.reference.measured", "parent.reference.published_base",
    *("current." + key for key in (*CURRENT_FIELDS, *FAMILY_KEYS)),
    *("current.observed." + key for key in CURRENT_FIELDS),
    "history.steps", "history.screened_steps", "history.screened_fraction",
    "history.complete_to_base", "history.latest_screened",
    *("history." + summary + "." + key for summary in (
        "mean", "max", "latest", "decay", "observed_fraction", "latest_observed",
    ) for key in HISTORY_FIELDS),
    *("history." + summary + "." + key for summary in ("mean", "latest")
      for key in FAMILY_KEYS),
)))
SELECTION_KEYS = frozenset({"selected_ids", "weights", "model", "feature_keys", "rationale"})
ROW_KEYS = frozenset({
    "example_id", "cell_id", "benchmark", "scientist_model", "split",
    "parent_reference", "reference_kind", "views",
})


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number, not missing or boolean")
    return float(value)


def _accuracy(value):
    number = _finite(value, "Accuracy")
    if not 0 <= number <= 1:
        raise ValueError("Accuracy must be in [0, 1]")
    return number


def _validate_row(row, *, training):
    if not isinstance(row, Mapping) or set(row) - ROW_KEYS:
        raise ValueError("Only input-only row fields are permitted; labels must be separate")
    for key in ("example_id", "cell_id", "benchmark"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError("Missing input identity: " + key)
    if training and row.get("split") != "train":
        raise ValueError("Only training rows may be selected for fitting")
    if not training and row.get("split") not in {"train", "validation", "test"}:
        raise ValueError("Query must declare its partition")
    reference = _accuracy(row.get("parent_reference"))
    kind = row.get("reference_kind")
    if kind not in {"measured_parent", "published_base"}:
        raise ValueError("Parent/reference accuracy must have a verified reference kind")
    views = row.get("views")
    if not isinstance(views, Mapping) or set(views) - {"parent", "current", "history"}:
        raise ValueError("Only whitelisted feature views are permitted")
    history = views.get("history")
    if not isinstance(history, Mapping) or set(history) != set(FEATURE_KEYS):
        raise ValueError("History view must have exactly the fixed 151 whitelisted features")
    for view in views.values():
        if not isinstance(view, Mapping) or set(view) - set(FEATURE_KEYS):
            raise ValueError("Non-whitelisted feature supplied")
        for key, value in view.items():
            _finite(value, "Feature " + key)
    if history["parent.accuracy"] != reference:
        raise ValueError("Parent accuracy feature must equal the supplied reference")
    if (history["parent.reference.measured"] != float(kind == "measured_parent")
            or history["parent.reference.published_base"] != float(kind == "published_base")):
        raise ValueError("Parent reference feature flags must agree with verified kind")


def validate_selection(selection, available_rows, query):
    """Validate a bounded actual-ID selection; return an independent plain dict.

    All available rows must be train-only, input-only rows from the query's
    benchmark, with no query-session overlap. Opaque aliases must be resolved by
    the caller before this boundary. Labels are intentionally not accepted here.
    """
    _validate_row(query, training=False)
    if not isinstance(selection, Mapping) or set(selection) != SELECTION_KEYS:
        raise ValueError("Selection must contain exactly the declared selection schema")
    if not isinstance(available_rows, (list, tuple)):
        raise ValueError("Available rows must be a sequence of training examples")  # noqa: TRY004
    available = {}
    for row in available_rows:
        _validate_row(row, training=True)
        key = row["example_id"]
        if key in available:
            raise ValueError("Duplicate available training identity")
        if row["benchmark"] != query["benchmark"]:
            raise ValueError("Selected/available examples must match the query benchmark")
        if key == query["example_id"] or row["cell_id"] == query["cell_id"]:
            raise ValueError("Query identity/session overlap with available training examples")
        available[key] = row
    ids = selection["selected_ids"]
    if not isinstance(ids, list) or not MIN_EXAMPLES <= len(ids) <= MAX_EXAMPLES:
        raise ValueError("Choose between 8 and 16 training examples")
    if any(not isinstance(key, str) for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("Selected example IDs must be unique strings")
    if set(ids) - set(available):
        raise ValueError("Selection contains unknown or held-out example IDs")
    if len({available[key]["cell_id"] for key in ids}) < MIN_SESSIONS:
        raise ValueError("Selection needs at least four independent training sessions")
    weights = selection["weights"]
    if not isinstance(weights, list) or len(weights) != len(ids):
        raise ValueError("One positive weight is required for every selected ID")
    weights = [_finite(weight, "Selection weight") for weight in weights]
    if any(not 0.25 <= weight <= 4.0 for weight in weights):
        raise ValueError("Selection weights must be in [0.25, 4]")
    model = selection["model"]
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError("Unknown lightweight model")
    keys = selection["feature_keys"]
    if (not isinstance(keys, list) or any(not isinstance(key, str) for key in keys)
            or len(keys) != len(set(keys)) or len(keys) > MAX_FEATURES
            or set(keys) - set(FEATURE_KEYS)):
        raise ValueError("Choose at most eight unique whitelisted feature names")
    if model != "weighted_median" and "parent.accuracy" not in keys:
        raise ValueError("Learned models require parent.accuracy among their features")
    rationale = selection["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Selection rationale must be a nonempty string")
    return {
        "selected_ids": list(ids), "weights": weights, "model": model,
        "feature_keys": list(keys), "rationale": rationale,
    }


def session_weights(rows, weights):
    """Agent weight / selected session count, normalized to global mean one."""
    counts = Counter(row["cell_id"] for row in rows)
    result = np.asarray([weight / counts[row["cell_id"]] for row, weight in zip(rows, weights)])
    result *= len(rows) / result.sum()
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Selection weight range is numerically degenerate")
    return result


def _fit_state(rows, labels, selection, weights):
    delta = np.asarray([labels[row["example_id"]]["delta_accuracy"] for row in rows])
    state = {"schema": "agent-local-fit-state-v1", "model": selection["model"],
             "feature_keys": list(selection["feature_keys"])}
    if selection["model"] == "weighted_median":
        order = np.argsort(delta, kind="stable")
        index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left")
        state["constant_delta"] = float(delta[order[min(int(index), len(order) - 1)]])
        return state
    keys = selection["feature_keys"]
    x = np.asarray([[row["views"]["history"][key] for key in keys] for row in rows])
    if selection["model"].startswith("ridge_"):
        # Exact constant columns must remain inactive even when weighted floating
        # point centering would leave tiny residuals in an otherwise constant LR.
        active = np.ptp(x, axis=0) != 0
        scaler = StandardScaler().fit(x, sample_weight=weights)
        scaler.scale_[~active] = 1.0
        transformed = scaler.transform(x)
        transformed[:, ~active] = 0.0
        alpha = int(selection["model"].split("_")[1])
        model = Ridge(alpha=alpha, solver="svd").fit(transformed, delta, sample_weight=weights)
        state.update({
            "alpha": alpha, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "active": active.tolist(), "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
        })
    else:
        model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=3, random_state=SEED)
        model.fit(x, delta, sample_weight=weights)
        tree = model.tree_
        state.update({
            "max_depth": 2, "min_samples_leaf": 3, "seed": SEED,
            "children_left": tree.children_left.tolist(), "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
            "value": tree.value[:, 0, 0].tolist(), "node_samples": tree.n_node_samples.tolist(),
        })
    return state


def predict_from_state(state, query):
    """Reproduce a fitted forecast from its JSON state, with no target labels."""
    _validate_row(query, training=False)
    if state.get("schema") != "agent-local-fit-state-v1" or state.get("model") not in MODELS:
        raise ValueError("Unknown serialized local model")
    model = state["model"]
    if model == "weighted_median":
        raw_delta = state["constant_delta"]
    else:
        keys = state["feature_keys"]
        if not keys or set(keys) - set(FEATURE_KEYS):
            raise ValueError("Serialized model contains non-whitelisted features")
        x = np.asarray([query["views"]["history"][key] for key in keys])
        if model.startswith("ridge_"):
            x = (x - np.asarray(state["mean"])) / np.asarray(state["scale"])
            x[~np.asarray(state["active"], dtype=bool)] = 0.0
            raw_delta = float(x @ np.asarray(state["coefficients"]) + state["intercept"])
        else:
            # sklearn trees consume float32 inputs; preserve that conversion at
            # prediction time to reproduce decisions immediately at thresholds.
            x = x.astype(np.float32)
            node = 0
            while state["children_left"][node] != -1:
                direction = "children_left" if x[state["feature"][node]] <= state["threshold"][node] else "children_right"
                node = state[direction][node]
            raw_delta = float(state["value"][node])
    raw_delta = _finite(raw_delta, "Fitted delta prediction")
    reference = _accuracy(query["parent_reference"])
    predicted = float(np.clip(reference + raw_delta, 0, 1))
    return {"predicted_accuracy": predicted, "predicted_delta": predicted - reference,
            "unclipped_predicted_accuracy": reference + raw_delta,
            "raw_predicted_delta": raw_delta, "clipped": predicted != reference + raw_delta}


def fit_predict(selected_train_rows, train_labels, query, selection):
    """Fit on exactly the chosen clean train rows; return prediction and audit state."""
    selection = validate_selection(selection, selected_train_rows, query)
    ids = [row["example_id"] for row in selected_train_rows]
    if ids != selection["selected_ids"]:
        raise ValueError("Training rows must have exactly selected_ids order")
    if not isinstance(train_labels, Mapping) or set(train_labels) != set(ids):
        raise ValueError("Provide labels only for the selected training rows; no held-out labels")
    for row in selected_train_rows:
        label = train_labels[row["example_id"]]
        if not isinstance(label, Mapping):
            raise ValueError("Missing official training label")  # noqa: TRY004
        target = _accuracy(label.get("accuracy"))
        delta = _finite(label.get("delta_accuracy"), "Delta label")
        if delta != target - row["parent_reference"]:
            raise ValueError("Delta label must equal valid target minus valid parent accuracy")
    weights = session_weights(selected_train_rows, selection["weights"])
    state = _fit_state(selected_train_rows, train_labels, selection, weights)
    forecast = predict_from_state(state, query)
    groups = sorted({row["cell_id"] for row in selected_train_rows})
    fold_records = []
    for group in groups:
        fit_indices = [i for i, row in enumerate(selected_train_rows) if row["cell_id"] != group]
        fit_rows = [selected_train_rows[i] for i in fit_indices]
        valid = [row for row in selected_train_rows if row["cell_id"] == group]
        fold_weights = session_weights(fit_rows, [selection["weights"][i] for i in fit_indices])
        fit_labels = {row["example_id"]: train_labels[row["example_id"]] for row in fit_rows}
        fold_state = _fit_state(fit_rows, fit_labels, selection, fold_weights)
        errors = [abs(predict_from_state(fold_state, row)["predicted_accuracy"]
                      - train_labels[row["example_id"]]["accuracy"]) for row in valid]
        fold_records.append({
            "held_out_training_session": group,
            "fit_ids": [row["example_id"] for row in fit_rows],
            "validation_ids": [row["example_id"] for row in valid],
            "mae": math.fsum(errors) / len(errors),
        })
    return {
        **forecast, "selection": selection, "model_state": state,
        "fit_metadata": {
            "training_examples": len(ids), "training_sessions": len(groups),
            "session_counts": dict(sorted(Counter(row["cell_id"] for row in selected_train_rows).items())),
            "normalized_sample_weights": weights.tolist(),
            "effective_sample_size": float(weights.sum() ** 2 / (weights @ weights)),
            "weight_policy": "Agent weight divided by selected count in that session, then normalized to global mean one; nonuniform weights preserve between-session relevance.",
            "feature_width": 0 if selection["model"] == "weighted_median" else len(selection["feature_keys"]),
            "target": "delta_accuracy", "scaling": "train-only weighted StandardScaler for Ridge; exact constant training columns inactive",
        },
        "conditional_training_diagnostics": {
            "description": "Leave-one-session-out MAE conditional on the already agent-selected subset/model; descriptive, not unbiased selection validation.",
            "equal_session_mae": math.fsum(fold["mae"] for fold in fold_records) / len(fold_records),
            "folds": fold_records,
        },
    }
