"""Outcome-blind matched neighborhoods and one fixed weighted-median estimator.

Matching uses prospective208-feature inputs and an audited positive-only treatment
sidecar. Recorded family tags are not claimed to be optimization objectives.
Unknown evidence never means absent. No labels enter candidate construction,
distances, session selection, weights or support checks. Under-supported matches
predict zero delta without filling across hard constraints.

The caller owns frozen source/sidecar provenance and complete training partition
membership. This module rejects nontraining rows and query-session overlap, even
when a neighborhood is empty. Validation-fold queries may have split=train only
after the caller removes their entire session from the training pool.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping

import numpy as np

from tools.outcome_prediction.wm_agent_broad_fit import validate_row
from tools.outcome_prediction.wm_agent_local_fit import _accuracy, _finite
from tools.outcome_prediction.wm_agent_recipe_data import FAMILIES
from tools.outcome_prediction.wm_matched_evidence import EVIDENCE_KEYS, validate_context

VARIANTS = ("all_rows", "state_only", "state_tag", "state_treatment_soft", "state_treatment_soft_shrunk")
TAG_KEYS = tuple("current.family." + name for name in ("sft", "rft", "rl", "merge", "decoding", "distill", "other"))
DOSE_FIELDS = (("learning_rate", 1e-5), ("epochs", 1.0), ("effective_batch_per_device", 8.0),
               ("sequence_length", 256.0), ("max_steps", 100.0))
PARENT_CALIPER, STATE_CALIPER, DISTANCE_CALIPER, MAX_SESSIONS = 0.20, 0.60, 0.60, 8
MIN_EXAMPLES, MIN_SESSIONS, MIN_SESSION_ESS = 6, 3, 2.5
MAX_AGENT_EXAMPLES, SHRINK_PRIOR_SESSIONS = 24, 4.0
SEED = "matched-neighborhood-v1-20260906"


def _hash(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _validate_pool(all_train, query):
    validate_row(query)
    if not isinstance(all_train, (list, tuple)):
        raise ValueError("Training pool must be an input-only row sequence")  # noqa: TRY004
    ids = set()
    for row in all_train:
        validate_row(row, training=True)
        if row["example_id"] in ids:
            raise ValueError("Duplicate training identity")
        ids.add(row["example_id"])
        if row["benchmark"] != query["benchmark"]:
            raise ValueError("All training rows must match the query benchmark")
        if row["example_id"] == query["example_id"] or row["cell_id"] == query["cell_id"]:
            raise ValueError("Training/query identity or session overlap")


def _unknown_context():
    step = {"status": "missing_recipe", "family_tokens": [], "objective": "unknown",
            "evidence": dict.fromkeys(EVIDENCE_KEYS)}
    return {"schema": "matched-treatment-evidence-v1", "current": copy.deepcopy(step),
            "immediate_parent": copy.deepcopy(step)}


def _validate_context(context):
    return copy.deepcopy(validate_context(context))


def _contexts(all_train, query, evidence):
    if evidence is None:
        return {row["example_id"]: _unknown_context() for row in [*all_train, query]}
    if not isinstance(evidence, Mapping):
        raise ValueError("Evidence must map example IDs to audited sidecars")  # noqa: TRY004
    result = {}
    for row in [*all_train, query]:
        value = evidence.get(row["example_id"])
        if not isinstance(value, Mapping):
            raise ValueError("Missing audited evidence for an input row")  # noqa: TRY004
        result[row["example_id"]] = _validate_context(value.get("context", value))
    return result


def _view(row):
    return row["views"]["history"]


def _mask(value):
    if value not in (0.0, 1.0):
        raise ValueError("Matching indicator must be a binary observed feature")
    return value == 1.0


def _tag(row):
    return frozenset(key for key in TAG_KEYS if _mask(_view(row)[key]))


def _jaccard(left, right):
    union = left | right
    return 1.0 - len(left & right) / len(union) if union else 0.5


def _dataset_distance(left, right, scope):
    a, b = _view(left), _view(right)
    if scope == "history":
        known = (a["recipe_data.history.known_identity_steps"] > 0
                 and b["recipe_data.history.known_identity_steps"] > 0)
        prefix = "recipe_data.history.any_family."
    else:
        prefix = "recipe_data." + scope + ".family."
        known = _mask(a["recipe_data." + scope + ".identity_known"]) and _mask(b["recipe_data." + scope + ".identity_known"])
    if not known:
        return 0.5
    return _jaccard(frozenset(f for f in FAMILIES if _mask(a[prefix + f])),
                    frozenset(f for f in FAMILIES if _mask(b[prefix + f])))


def _dose_distance(left, right, *, parent=False):
    a, b = _view(left), _view(right)
    value_prefix = "history.latest.config." if parent else "current.config."
    mask_prefix = "history.latest_observed.config." if parent else "current.observed.config."
    parts = {}
    for name, scale in DOSE_FIELDS:
        x, y = a[value_prefix + name], b[value_prefix + name]
        if not (_mask(a[mask_prefix + name]) and _mask(b[mask_prefix + name])) or min(x, y) < 0:
            parts[name] = 0.5
        else:
            # log1p(x/scale) computed without overflow for finite prospective values.
            transform = lambda value, unit=scale: math.log(value + unit) - math.log(unit)
            parts[name] = min(abs(transform(x) - transform(y)) / math.log(4), 1.0)
    return math.fsum(parts.values()) / len(parts), parts


def _objective_distance(left, right, scope):
    a, b = left[scope]["objective"], right[scope]["objective"]
    return 0.5 if "unknown" in (a, b) else float(a != b)


def _objective_compatible(left, right):
    a, b = left["current"]["objective"], right["current"]["objective"]
    return "unknown" in (a, b) or a == b


def distance_components(row, query, row_context, query_context):
    """Fixed missing-aware distances; only prior recipes enter the state part."""
    parent = min(abs(row["parent_reference"] - query["parent_reference"]) / PARENT_CALIPER, 1.0)
    a, b = _view(row), _view(query)
    n, m = a["recipe_data.history.steps"], b["recipe_data.history.steps"]
    if min(n, m) < 0:
        raise ValueError("History length must be nonnegative")
    depth = min(abs(math.log1p(n) - math.log1p(m)) / math.log(4), 1.0)
    parent_dataset = _dataset_distance(row, query, "parent")
    parent_dose, parent_parts = _dose_distance(row, query, parent=True)
    ancestor_dataset = _dataset_distance(row, query, "history")
    parent_objective = _objective_distance(row_context, query_context, "immediate_parent")
    history = (0.30 * parent_dataset + 0.30 * parent_dose + 0.20 * ancestor_dataset
               + 0.10 * parent_objective + 0.10 * depth)
    if row["reference_kind"] == query["reference_kind"] == "published_base" and n == m == 0:
        history = 0.0
    state = 0.70 * parent + 0.30 * history
    dataset = _dataset_distance(row, query, "current")
    dose, dose_parts = _dose_distance(row, query)
    tag = _jaccard(_tag(row), _tag(query))
    positive = {key: 0.0 if row_context["current"]["evidence"][key] is True
                and query_context["current"]["evidence"][key] is True else 0.5 for key in EVIDENCE_KEYS}
    treatment = (tag + math.fsum(positive.values()) / len(positive)) / 2.0
    if _objective_distance(row_context, query_context, "current") == 0.5:
        treatment = max(treatment, 0.5)
    total = 0.50 * state + 0.20 * dataset + 0.20 * dose + 0.10 * treatment
    return {"parent_accuracy": parent, "history": history, "state": state,
            "history_parts": {"parent_dataset": parent_dataset, "parent_dose": parent_dose,
                              "ancestor_dataset": ancestor_dataset, "parent_objective": parent_objective,
                              "depth": depth, "parent_dose_parts": parent_parts},
            "current_dataset": dataset, "current_dose": dose, "dose_parts": dose_parts,
            "recorded_tag": tag, "positive_evidence_parts": positive,
            "treatment_signature": treatment, "state_treatment": total}


def _base_compatible(row, query):
    return (row["reference_kind"] == query["reference_kind"]
            and abs(row["parent_reference"] - query["parent_reference"]) <= PARENT_CALIPER + 1e-12)


def candidate_rows(all_train, query, evidence=None):
    """Agent candidates: shared state caliper/objective compatibility, no joint cut."""
    _validate_pool(all_train, query)
    contexts = _contexts(all_train, query, evidence)
    return [row for row in all_train if _base_compatible(row, query)
            and _objective_compatible(contexts[row["example_id"]], contexts[query["example_id"]])
            and distance_components(row, query, contexts[row["example_id"]], contexts[query["example_id"]])["state"] <= STATE_CALIPER + 1e-12]


def _effective(rows, relevance):
    if not rows:
        return []
    counts = Counter(row["cell_id"] for row in rows)
    weights = [weight / counts[row["cell_id"]] for row, weight in zip(rows, relevance)]
    total = math.fsum(weights)
    return [weight * len(rows) / total for weight in weights]


def _support(rows, weights):
    totals = {}
    for row, weight in zip(rows, weights):
        totals[row["cell_id"]] = totals.get(row["cell_id"], 0.0) + weight
    total = math.fsum(weights)
    row_ess = total ** 2 / math.fsum(w ** 2 for w in weights) if weights else 0.0
    session_ess = total ** 2 / math.fsum(w ** 2 for w in totals.values()) if totals else 0.0
    reasons = []
    if len(rows) < MIN_EXAMPLES:
        reasons.append("fewer_than_6_examples")
    if len(totals) < MIN_SESSIONS:
        reasons.append("fewer_than_3_sessions")
    if session_ess + 1e-12 < MIN_SESSION_ESS:
        reasons.append("session_ess_below_2.5")
    return {"supported": not reasons, "reason": ";".join(reasons) if reasons else "supported",
            "reasons": reasons, "examples": len(rows), "sessions": len(totals),
            "row_effective_sample_size": row_ess, "session_effective_sample_size": session_ess,
            "session_total_weights": dict(sorted(totals.items()))}


def _input_signature(all_train, query):
    def safe(row):
        return {key: row[key] for key in ("example_id", "cell_id", "benchmark", "split",
                                         "parent_reference", "reference_kind", "views")}
    return _hash({"train": [safe(row) for row in sorted(all_train, key=lambda row: row["example_id"])], "query": safe(query)})


def _make_neighborhood(all_train, query, chosen, relevance, distances, components, contexts, variant, counts):
    weights = _effective(chosen, relevance)
    value = {"schema": "matched-neighborhood-v1", "variant": variant,
             "query_id": query["example_id"], "input_signature": _input_signature(all_train, query),
             "selected_ids": [row["example_id"] for row in chosen], "raw_relevance": relevance,
             "effective_weights": weights, "distances": distances, "distance_components": components,
             "support": _support(chosen, weights), "candidate_counts": counts,
             "matching_evidence": {row["example_id"]: contexts[row["example_id"]] for row in [*chosen, query]},
             "weight_policy": "Relevance divided by retained examples per session; normalized to mean one. No padding across hard/caliper constraints."}
    value["construction_sha256"] = _hash(value)
    return value


def neighborhood(all_train, query, variant="state_treatment_soft", evidence=None):
    """Construct a frozen, label-blind rules neighborhood and support diagnostic."""
    _validate_pool(all_train, query)
    if variant not in VARIANTS:
        raise ValueError("Unknown predeclared matching variant")
    contexts = _contexts(all_train, query, evidence)
    q_context = contexts[query["example_id"]]
    counts = {"training_examples": len(all_train), "hard_compatible_examples": 0, "state_compatible_examples": 0,
              "distance_eligible_examples": 0, "distance_eligible_sessions": 0}
    if variant == "all_rows":
        chosen = sorted(all_train, key=lambda row: row["example_id"])
        counts.update(hard_compatible_examples=len(chosen), state_compatible_examples=len(chosen), distance_eligible_examples=len(chosen),
                      distance_eligible_sessions=len({row["cell_id"] for row in chosen}))
        return _make_neighborhood(all_train, query, chosen, [1.0] * len(chosen), [0.0] * len(chosen),
                                  [{} for _ in chosen], contexts, variant, counts)
    candidates = []
    for row in all_train:
        if not _base_compatible(row, query):
            continue
        if variant == "state_tag" and _tag(row) != _tag(query):
            continue
        if variant != "state_only" and not _objective_compatible(contexts[row["example_id"]], q_context):
            continue
        counts["hard_compatible_examples"] += 1
        parts = distance_components(row, query, contexts[row["example_id"]], q_context)
        if parts["state"] > STATE_CALIPER + 1e-12:
            continue
        counts["state_compatible_examples"] += 1
        distance = parts["state"] if variant == "state_only" else parts["state_treatment"]
        if distance <= DISTANCE_CALIPER + 1e-12:
            candidates.append((row, distance, parts))
    counts["distance_eligible_examples"] = len(candidates)
    session_min = {}
    for row, distance, _ in candidates:
        session_min[row["cell_id"]] = min(session_min.get(row["cell_id"], math.inf), distance)
    counts["distance_eligible_sessions"] = len(session_min)
    sessions = set(sorted(session_min, key=lambda group: (session_min[group], _hash(SEED + "|" + group)))[:MAX_SESSIONS])
    retained = sorted((item for item in candidates if item[0]["cell_id"] in sessions), key=lambda item: item[0]["example_id"])
    chosen, distances, components = ([item[i] for item in retained] for i in (0, 1, 2))
    relevance = [math.exp(-3.0 * distance) for distance in distances]
    return _make_neighborhood(all_train, query, chosen, relevance, distances, components, contexts, variant, counts)


def selected_neighborhood(all_train, query, selected_ids, weights, evidence=None):
    """Validate an agent's choice from the hard pool; an empty choice is supported as abstention."""
    candidates = candidate_rows(all_train, query, evidence)
    if (not isinstance(selected_ids, list) or len(selected_ids) > MAX_AGENT_EXAMPLES
            or any(not isinstance(key, str) for key in selected_ids) or len(selected_ids) != len(set(selected_ids))):
        raise ValueError("Choose at most24 unique candidate IDs, or an empty unsupported selection")
    by_id = {row["example_id"]: row for row in candidates}
    if set(selected_ids) - set(by_id):
        raise ValueError("Agent selection violates hard candidate constraints")
    if not isinstance(weights, list) or len(weights) != len(selected_ids):
        raise ValueError("One weight is required per chosen candidate")
    weights = [_finite(weight, "Agent relevance") for weight in weights]
    if any(not 0.5 <= weight <= 2 for weight in weights):
        raise ValueError("Agent relevance must be in [0.5,2]")
    contexts = _contexts(all_train, query, evidence)
    chosen = [by_id[key] for key in selected_ids]
    parts = [distance_components(row, query, contexts[row["example_id"]], contexts[query["example_id"]]) for row in chosen]
    counts = {"training_examples": len(all_train), "hard_compatible_examples": len(candidates),
              "state_compatible_examples": len(candidates), "agent_selected_examples": len(chosen),
              "shared_state_caliper": STATE_CALIPER, "joint_distance_caliper_applied": False}
    return _make_neighborhood(all_train, query, chosen, weights, [p["state_treatment"] for p in parts],
                              parts, contexts, "agent_selected", counts)


def fit_from_neighborhood(all_train, train_labels, query, matched, shrink=False):
    """One fixed delta median; consumes only clean TRAIN labels after matching."""
    _validate_pool(all_train, query)
    ids = {row["example_id"] for row in all_train}
    if not isinstance(train_labels, Mapping) or set(train_labels) != ids:
        raise ValueError("Labels must match exactly the clean training pool; never query labels")
    for row in all_train:
        label = train_labels[row["example_id"]]
        if not isinstance(label, Mapping):
            raise ValueError("Missing official training label")  # noqa: TRY004
        target = _accuracy(label.get("accuracy"))
        delta = _finite(label.get("delta_accuracy"), "Delta label")
        if delta != target - row["parent_reference"]:
            raise ValueError("Delta label must equal valid target minus valid parent accuracy")
    if type(shrink) is not bool:
        raise ValueError("Shrink choice must be a predeclared boolean")
    if not isinstance(matched, Mapping) or matched.get("schema") != "matched-neighborhood-v1":
        raise ValueError("Invalid neighborhood schema")
    sealed = {key: value for key, value in matched.items() if key != "construction_sha256"}
    if (matched.get("construction_sha256") != _hash(sealed) or matched["input_signature"] != _input_signature(all_train, query)
            or matched["query_id"] != query["example_id"]):
        raise ValueError("Neighborhood/input signature changed")
    selected = matched["selected_ids"]
    if not isinstance(selected, list) or len(selected) != len(set(selected)) or set(selected) - ids:
        raise ValueError("Invalid neighborhood training IDs")
    by_id = {row["example_id"]: row for row in all_train}
    chosen = [by_id[key] for key in selected]
    contexts = _contexts(chosen, query, matched["matching_evidence"])
    weights = matched["effective_weights"]
    relevance = matched["raw_relevance"]
    if (len(weights) != len(chosen) or len(relevance) != len(chosen)
            or any(_finite(value, "Neighborhood weight") <= 0 for value in [*weights, *relevance])
            or weights != _effective(chosen, relevance) or matched["support"] != _support(chosen, weights)):
        raise ValueError("Neighborhood weights/support changed")
    variant = matched["variant"]
    if variant not in (*VARIANTS, "agent_selected"):
        raise ValueError("Unknown neighborhood variant")
    if len(matched["distances"]) != len(chosen) or len(matched["distance_components"]) != len(chosen):
        raise ValueError("Neighborhood distance alignment changed")
    if variant == "all_rows" and selected != sorted(ids):
        raise ValueError("The all-rows baseline cannot omit training examples")
    if variant == "agent_selected" and (len(chosen) > MAX_AGENT_EXAMPLES or any(not 0.5 <= w <= 2 for w in relevance)):
        raise ValueError("Agent neighborhood exceeds frozen selection/weight bounds")
    if variant not in {"all_rows", "agent_selected"} and len({row["cell_id"] for row in chosen}) > MAX_SESSIONS:
        raise ValueError("Rules neighborhood exceeds the session cap")
    for index, row in enumerate(chosen):
        if variant != "all_rows" and not _base_compatible(row, query):
            raise ValueError("Neighborhood violates hard state constraints")
        if variant == "state_tag" and _tag(row) != _tag(query):
            raise ValueError("Neighborhood violates recorded-family constraint")
        if variant not in {"all_rows", "state_only"} and not _objective_compatible(contexts[row["example_id"]], contexts[query["example_id"]]):
            raise ValueError("Neighborhood violates known-objective compatibility")
        if variant == "all_rows":
            if relevance[index] != 1.0 or matched["distances"][index] != 0.0:
                raise ValueError("All-row baseline weights must be session-uniform")
            continue
        parts = distance_components(row, query, contexts[row["example_id"]], contexts[query["example_id"]])
        if parts["state"] > STATE_CALIPER + 1e-12:
            raise ValueError("Neighborhood violates shared state caliper")
        distance = parts["state"] if variant == "state_only" else parts["state_treatment"]
        if parts != matched["distance_components"][index] or distance != matched["distances"][index]:
            raise ValueError("Neighborhood distances fail input-only replay")
        if variant != "agent_selected" and (distance > DISTANCE_CALIPER + 1e-12 or relevance[index] != math.exp(-3.0 * distance)):
            raise ValueError("Rules neighborhood violates distance caliper/weight rule")
    supported = matched["support"]["supported"]
    raw_delta = 0.0
    if supported:
        deltas = np.asarray([train_labels[key]["delta_accuracy"] for key in selected])
        numeric_weights = np.asarray(weights)
        order = np.argsort(deltas, kind="stable")
        index = np.searchsorted(np.cumsum(numeric_weights[order]), numeric_weights.sum() / 2, side="left")
        raw_delta = float(deltas[order[min(int(index), len(order) - 1)]])
    ess = matched["support"]["session_effective_sample_size"]
    factor = ess / (ess + SHRINK_PRIOR_SESSIONS) if shrink and supported else 1.0
    delta = raw_delta * factor
    reference = _accuracy(query["parent_reference"])
    predicted = float(np.clip(reference + delta, 0, 1))
    return {"predicted_accuracy": predicted, "predicted_delta": predicted - reference,
            "raw_median_delta": raw_delta, "unclipped_predicted_accuracy": reference + delta,
            "clipped": predicted != reference + delta, "neighborhood": copy.deepcopy(dict(matched)),
            "fallback": {"used": not supported, "prediction": "zero_delta" if not supported else None,
                         "reason": matched["support"]["reason"] if not supported else None},
            "shrink": {"enabled": shrink, "factor": factor, "prior_sessions": SHRINK_PRIOR_SESSIONS},
            "estimator": "Fixed session-adjusted weighted median delta; no outcome-dependent neighborhood/model selection."}


def fit_predict(all_train, train_labels, query, variant="state_treatment_soft", evidence=None):
    matched = neighborhood(all_train, query, variant, evidence)
    return fit_from_neighborhood(all_train, train_labels, query, matched,
                                 shrink=variant == "state_treatment_soft_shrunk")
