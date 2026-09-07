"""Synthetic-only outcome-blind matching, support and fixed delta-estimator tests."""

import copy
import json

import pytest

from tools.outcome_prediction import wm_matched_fit as matched
from tools.outcome_prediction.wm_agent_broad_fit import FEATURE_KEYS
from tools.outcome_prediction.wm_agent_recipe_data import extract as data_extract
from tools.outcome_prediction.wm_matched_evidence import extract as evidence_extract


def fixture(n=20, per_session=2):
    model_input = {"plan": {"setup": {"method": {"family": "sft"},
                                      "data": [{"source": "openai/gsm8k", "selection": "Keep only correct solutions"}]}}}
    example = {"model_input": model_input, "history": [{"recipe_status": "screened", "model_input": model_input}],
               "history_complete_to_base": True}
    data, evidence = data_extract(example), evidence_extract(example)
    rows, labels, contexts = [], {}, {}
    for i in range(n):
        reference = 0.5 + (i % 3) / 100
        values = dict.fromkeys(FEATURE_KEYS, 0.0)
        values.update(data["features"])
        values.update({"parent.accuracy": reference, "parent.reference.measured": 1.0,
                       "current.family.sft": 1.0, "history.latest.family.sft": 1.0})
        for name, scale in matched.DOSE_FIELDS:
            values["current.config." + name] = scale
            values["current.observed.config." + name] = 1.0
            values["history.latest.config." + name] = scale
            values["history.latest_observed.config." + name] = 1.0
        identity = f"train-{i // per_session}/exp-{i}"
        rows.append({"example_id": identity, "cell_id": f"train-{i // per_session}",
                     "benchmark": "synthetic", "split": "train", "parent_reference": reference,
                     "reference_kind": "measured_parent", "views": {"history": values}})
        target = reference + 0.05 + (i % 4) / 100
        labels[identity] = {"accuracy": target, "delta_accuracy": target - reference}
        contexts[identity] = copy.deepcopy(evidence)
    query = copy.deepcopy(rows[0]) if rows else fixture(1)[0][0]
    query.update(example_id="heldout/query", cell_id="heldout", split="test")
    contexts[query["example_id"]] = copy.deepcopy(evidence)
    return rows, labels, query, contexts


@pytest.mark.parametrize("variant", matched.VARIANTS)
def test_all_fixed_variants_deterministic_serializable_and_replayable(variant):
    rows, labels, query, evidence = fixture()
    before = copy.deepcopy((rows, labels, query, evidence))
    result = matched.fit_predict(rows, labels, query, variant, evidence)
    assert result == matched.fit_predict(rows, labels, query, variant, evidence)
    saved = json.loads(json.dumps(result, allow_nan=False))
    assert result == matched.fit_from_neighborhood(rows, labels, query, saved["neighborhood"],
                                                   shrink=variant.endswith("_shrunk"))
    assert result["neighborhood"]["support"]["supported"]
    assert 0 <= result["predicted_accuracy"] <= 1
    assert result["predicted_delta"] == result["predicted_accuracy"] - query["parent_reference"]
    assert before == (rows, labels, query, evidence)


def test_matching_api_has_no_labels_and_uses_same_neighborhood_if_labels_change():
    rows, labels, query, evidence = fixture()
    first = matched.fit_predict(rows, labels, query, "state_treatment_soft", evidence)
    changed = {row["example_id"]: {"accuracy": 0.0, "delta_accuracy": -row["parent_reference"]} for row in rows}
    second = matched.fit_predict(rows, changed, query, "state_treatment_soft", evidence)
    assert first["neighborhood"] == second["neighborhood"]
    assert first["predicted_accuracy"] != second["predicted_accuracy"]
    assert second["predicted_delta"] < 0  # no forced improvement


def test_strict_recorded_tags_differ_from_semantic_supervised_objective():
    rows, _, query, evidence = fixture(6)
    for row in rows:
        row["views"]["history"]["current.family.sft"] = 0.0
        row["views"]["history"]["current.family.rft"] = 1.0
        evidence[row["example_id"]]["context"]["current"]["family_tokens"] = ["rft"]
    strict = matched.neighborhood(rows, query, "state_tag", evidence)
    soft = matched.neighborhood(rows, query, "state_treatment_soft", evidence)
    assert strict["selected_ids"] == []
    assert len(soft["selected_ids"]) == 6
    assert soft["support"]["supported"]


def test_state_only_uses_prior_recipe_not_current_treatment():
    rows, _, query, evidence = fixture()
    original = matched.neighborhood(rows, query, "state_only", evidence)
    query["views"]["history"]["current.family.sft"] = 0.0
    query["views"]["history"]["current.family.rl"] = 1.0
    query["views"]["history"]["current.config.learning_rate"] = 0.1
    query["views"]["history"]["recipe_data.current.family.gsm8k"] = 0.0
    query["views"]["history"]["recipe_data.current.family.metamathqa"] = 1.0
    evidence[query["example_id"]]["context"]["current"]["objective"] = "unknown"
    evidence[query["example_id"]]["context"]["current"]["family_tokens"] = ["rl"]
    current_changed = matched.neighborhood(rows, query, "state_only", evidence)
    assert original["selected_ids"] == current_changed["selected_ids"]
    assert original["effective_weights"] == current_changed["effective_weights"]
    query["views"]["history"]["history.latest.config.learning_rate"] = 0.1
    parent_changed = matched.neighborhood(rows, query, "state_only", evidence)
    assert original["distances"] != parent_changed["distances"]


def test_positive_only_evidence_unknown_is_not_absence_or_exact_match():
    rows, _, query, evidence = fixture(6)
    context_a = evidence[rows[0]["example_id"]]["context"]
    context_b = evidence[query["example_id"]]["context"]
    context_a["current"]["evidence"]["wrong_attempts"] = True
    context_b["current"]["evidence"]["wrong_attempts"] = None
    parts = matched.distance_components(rows[0], query, context_a, context_b)
    assert parts["positive_evidence_parts"]["wrong_attempts"] == 0.5
    assert parts["positive_evidence_parts"]["correctness_selection"] == 0
    context_b["current"]["evidence"]["wrong_attempts"] = True
    assert matched.distance_components(rows[0], query, context_a, context_b)["positive_evidence_parts"]["wrong_attempts"] == 0
    context_b["current"]["evidence"]["wrong_attempts"] = False
    with pytest.raises(ValueError, match="True or None"):
        matched.neighborhood(rows, query, "state_treatment_soft", evidence)


def test_legacy_correctness_zero_is_never_treated_as_absence():
    rows, _, query, evidence = fixture()
    before = matched.neighborhood(rows, query, "state_treatment_soft", evidence)
    query["views"]["history"]["current.data.policy.evidence.correctness_filter"] = 0.0
    query["views"]["history"]["current.observed.data.policy.evidence.correctness_filter"] = 1.0
    after = matched.neighborhood(rows, query, "state_treatment_soft", evidence)
    assert before["distances"] == after["distances"]
    assert before["effective_weights"] == after["effective_weights"]


def test_unknown_objective_is_allowed_with_penalty():
    rows, _, query, evidence = fixture(6)
    evidence[query["example_id"]]["context"]["current"]["objective"] = "unknown"
    evidence[query["example_id"]]["context"]["current"]["family_tokens"] = []
    result = matched.neighborhood(rows, query, "state_treatment_soft", evidence)
    assert len(matched.candidate_rows(rows, query, evidence)) == 6
    assert all(item["treatment_signature"] >= 0.5 for item in result["distance_components"])


def test_known_base_empty_history_is_zero_history_distance():
    rows, _, query, evidence = fixture(6)
    for row in [*rows, query]:
        row["reference_kind"] = "published_base"
        row["views"]["history"]["parent.reference.measured"] = 0.0
        row["views"]["history"]["parent.reference.published_base"] = 1.0
        row["views"]["history"]["recipe_data.history.steps"] = 0.0
    result = matched.neighborhood(rows, query, "state_only", evidence)
    assert all(item["history"] == 0 for item in result["distance_components"])


@pytest.mark.parametrize("rule", ["reference_kind", "parent_caliper", "distance_caliper"])
def test_no_padding_or_distant_session_sibling_retention(rule):
    rows, _, query, evidence = fixture(8)
    excluded = rows[-1]
    if rule == "reference_kind":
        excluded["reference_kind"] = "published_base"
        excluded["views"]["history"]["parent.reference.measured"] = 0.0
        excluded["views"]["history"]["parent.reference.published_base"] = 1.0
    else:
        excluded["parent_reference"] = 0.71 if rule == "parent_caliper" else 0.70
        excluded["views"]["history"]["parent.accuracy"] = excluded["parent_reference"]
    result = matched.neighborhood(rows, query, "state_only", evidence)
    assert excluded["example_id"] not in result["selected_ids"]
    assert rows[-2]["example_id"] in result["selected_ids"]  # near sibling retained, distant sibling not filled
    if rule == "distance_caliper":
        assert excluded not in matched.candidate_rows(rows, query, evidence)  # shared state caliper applies to agents too


def test_treatment_joint_distance_never_readmits_shared_state_rejection():
    rows, _, query, evidence = fixture(8)
    excluded = rows[-1]
    excluded["parent_reference"] = 0.70
    excluded["views"]["history"]["parent.accuracy"] = 0.70
    parts = matched.distance_components(excluded, query, evidence[excluded["example_id"]]["context"],
                                        evidence[query["example_id"]]["context"])
    assert parts["state"] > matched.STATE_CALIPER
    assert parts["state_treatment"] < matched.DISTANCE_CALIPER
    for variant in matched.VARIANTS[1:]:
        result = matched.neighborhood(rows, query, variant, evidence)
        assert excluded["example_id"] not in result["selected_ids"]
    with pytest.raises(ValueError, match="hard candidate constraints"):
        matched.selected_neighborhood(rows, query, [excluded["example_id"]], [1.0], evidence)


def test_rules_limit_to_eight_sessions_deterministically_even_under_input_permutation():
    rows, _, query, evidence = fixture(30)
    first = matched.neighborhood(rows, query, "state_treatment_soft", evidence)
    second = matched.neighborhood(list(reversed(rows)), query, "state_treatment_soft", evidence)
    assert first == second
    assert first["support"]["sessions"] == 8


@pytest.mark.parametrize("case", ["too_few_rows", "too_few_sessions", "low_ess", "empty"])
def test_unsupported_selection_keeps_output_coverage_with_zero_delta(case):
    rows, labels, query, evidence = fixture(12)
    if case == "too_few_rows":
        selected, weights = rows[:5], [1.0] * 5
    elif case == "too_few_sessions":
        selected, weights = rows[:8], [1.0] * 8
        for i, row in enumerate(rows):
            row["cell_id"] = f"train-session-{i % 2}"
    elif case == "low_ess":
        selected, weights = rows[:6], [2.0, 2.0, 0.5, 0.5, 0.5, 0.5]
    else:
        selected, weights = [], []
    neighborhood = matched.selected_neighborhood(rows, query, [row["example_id"] for row in selected], weights, evidence)
    result = matched.fit_from_neighborhood(rows, labels, query, neighborhood)
    assert not neighborhood["support"]["supported"]
    assert result["fallback"]["used"]
    assert result["predicted_delta"] == 0.0
    assert result["predicted_accuracy"] == query["parent_reference"]


def test_session_weighting_caps_repeated_cards_and_support_gate_uses_ess():
    rows, _, query, evidence = fixture(8)
    rows[0]["cell_id"] = rows[2]["cell_id"]
    selected = matched.selected_neighborhood(rows, query, [row["example_id"] for row in rows], [1.0] * len(rows), evidence)
    sums = selected["support"]["session_total_weights"]
    assert list(sums.values()) == pytest.approx([len(rows) / len(sums)] * len(sums))
    assert selected["support"]["session_effective_sample_size"] == pytest.approx(len(sums))


def test_shrink_changes_only_supported_median_towards_zero():
    rows, labels, query, evidence = fixture()
    plain = matched.fit_predict(rows, labels, query, "state_treatment_soft", evidence)
    shrunk = matched.fit_predict(rows, labels, query, "state_treatment_soft_shrunk", evidence)
    assert plain["neighborhood"]["selected_ids"] == shrunk["neighborhood"]["selected_ids"]
    assert plain["neighborhood"]["effective_weights"] == shrunk["neighborhood"]["effective_weights"]
    ess = plain["neighborhood"]["support"]["session_effective_sample_size"]
    assert shrunk["shrink"]["factor"] == pytest.approx(ess / (ess + 4))
    assert shrunk["predicted_delta"] == pytest.approx(plain["raw_median_delta"] * ess / (ess + 4))


@pytest.mark.parametrize("value", [None, True, "0.5", float("nan"), float("inf"), -0.1, 1.1])
@pytest.mark.parametrize("owner", ["target", "parent", "query"])
def test_unknown_or_invalid_accuracies_rejected_even_if_no_support(value, owner):
    rows, labels, query, evidence = fixture(4)
    if owner == "target":
        labels[rows[0]["example_id"]]["accuracy"] = value
    elif owner == "parent":
        rows[0]["parent_reference"] = value
    else:
        query["parent_reference"] = value
    with pytest.raises(ValueError):
        matched.fit_predict(rows, labels, query, "state_only", evidence)


@pytest.mark.parametrize("invalid", ["test_row", "session_overlap", "query_label", "extra_label", "missing_delta", "benchmark", "missing_evidence"])
def test_input_label_and_session_boundaries(invalid):
    rows, labels, query, evidence = fixture()
    if invalid == "test_row":
        rows[0]["split"] = "test"
    elif invalid == "session_overlap":
        query["cell_id"] = rows[0]["cell_id"]
    elif invalid == "query_label":
        query["accuracy"] = 0.9
    elif invalid == "extra_label":
        labels[query["example_id"]] = {"accuracy": 0.9, "delta_accuracy": 0.4}
    elif invalid == "missing_delta":
        labels[rows[0]["example_id"]].pop("delta_accuracy")
    elif invalid == "benchmark":
        rows[0]["benchmark"] = "different"
    else:
        evidence.pop(rows[0]["example_id"])
    with pytest.raises(ValueError):
        matched.fit_predict(rows, labels, query, "state_treatment_soft", evidence)


@pytest.mark.parametrize("invalid", ["duplicate", "unknown", "too_many", "zero", "low", "high", "missing_weights"])
def test_agent_selection_constraints(invalid):
    rows, _, query, evidence = fixture(30)
    ids, weights = [row["example_id"] for row in rows[:12]], [1.0] * 12
    if invalid == "duplicate":
        ids[0] = ids[1]
    elif invalid == "unknown":
        ids[0] = query["example_id"]
    elif invalid == "too_many":
        ids, weights = [row["example_id"] for row in rows[:25]], [1.0] * 25
    elif invalid == "missing_weights":
        weights.pop()
    else:
        weights[0] = {"zero": 0, "low": 0.49, "high": 2.01}[invalid]
    with pytest.raises(ValueError):
        matched.selected_neighborhood(rows, query, ids, weights, evidence)


def test_neighborhood_tampering_and_changed_inputs_rejected():
    rows, labels, query, evidence = fixture()
    selected = matched.neighborhood(rows, query, "state_treatment_soft", evidence)
    corrupt = copy.deepcopy(selected)
    corrupt["effective_weights"][0] = 999.0
    with pytest.raises(ValueError, match="signature"):
        matched.fit_from_neighborhood(rows, labels, query, corrupt)
    query["views"]["history"]["current.config.epochs"] = 10.0
    with pytest.raises(ValueError, match="signature"):
        matched.fit_from_neighborhood(rows, labels, query, selected)


def test_cv_query_train_split_allowed_after_whole_session_removal():
    rows, labels, _, evidence = fixture()
    query = rows[0]
    training = [row for row in rows if row["cell_id"] != query["cell_id"]]
    training_labels = {row["example_id"]: labels[row["example_id"]] for row in training}
    result = matched.fit_predict(training, training_labels, query, "state_treatment_soft", evidence)
    assert result["neighborhood"]["support"]["supported"]
    assert not set(result["neighborhood"]["selected_ids"]) & {row["example_id"] for row in rows if row["cell_id"] == query["cell_id"]}


def test_actual_zero_scores_valid_and_optional_evidence_remains_unknown():
    rows, labels, query, _ = fixture(6)
    for row in rows:
        labels[row["example_id"]] = {"accuracy": 0.0, "delta_accuracy": -row["parent_reference"]}
    result = matched.fit_predict(rows, labels, query, "all_rows")
    assert result["predicted_accuracy"] == pytest.approx(0.0)
    assert result["neighborhood"]["support"]["supported"]
    assert all(context["current"]["objective"] == "unknown" for context in result["neighborhood"]["matching_evidence"].values())
