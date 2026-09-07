"""Synthetic full-pool prompt and spec boundaries; no fits, calls or artifacts."""

import copy
import json

import pytest

from tools.outcome_prediction import wm_agent_broad_benchmark as benchmark
from tools.outcome_prediction.wm_agent_broad_fit import FEATURE_KEYS
from tools.outcome_prediction.wm_agent_recipe_data import extract


def fixture():
    rows, labels, evidence = [], {}, {}
    for i in range(51):
        reference = 0.1 + i / 1000
        source = "openai/gsm8k" if i % 2 else "meta-math/MetaMathQA"
        item = extract({"model_input": {"plan": {"setup": {"data": [{"source": source}]}}, "code": []},
                        "history": [], "history_complete_to_base": True,
                        "accuracy": 0.987654321234, "private_identifier": "do-not-send-raw-identifier"})
        values = dict.fromkeys(FEATURE_KEYS, 0.0)
        values.update(item["features"])
        values.update({"parent.accuracy": reference, "parent.reference.measured": 1.0,
                       "current.config.epochs": float(i % 4)})
        identity = f"private-training-session-{i // 3}/private-experiment-{i}"
        rows.append({"example_id": identity, "cell_id": f"private-training-session-{i // 3}",
                     "benchmark": "gsm8k", "split": "train", "parent_reference": reference,
                     "reference_kind": "measured_parent", "scientist_model": "private-scientist-model",
                     "views": {"history": values}})
        target = i / 100
        labels[identity] = {"accuracy": target, "delta_accuracy": target - reference}
        item["provenance"]["private_test_score"] = "private-provenance-not-prompted"
        evidence[identity] = item
    query = copy.deepcopy(rows[0])
    query.update(example_id="private-heldout/query", cell_id="private-heldout", split="test")
    query["views"]["history"]["current.config.epochs"] = 100.0
    evidence[query["example_id"]] = copy.deepcopy(evidence[rows[0]["example_id"]])
    aliases = benchmark.old.aliases_for(rows)
    response = {"weight_overrides": {}, "target": "accuracy", "model": "ridge_10",
                "feature_keys": ["parent.accuracy", "recipe_data.current.family.gsm8k"],
                "rationale": "Private selector reasoning that must not enter direct inference."}
    return rows, labels, query, aliases, evidence, response, list(FEATURE_KEYS)


def payload_fixture():
    rows, labels, query, aliases, evidence, response, order = fixture()
    payload = benchmark.prompt_payload(query, rows, labels, aliases, evidence, order)
    return rows, labels, query, aliases, evidence, response, order, payload


def test_shared_context_identical_across_queries_and_serialized_before_query():
    rows, labels, query, aliases, evidence, _, order = fixture()
    second = copy.deepcopy(query)
    second.update(example_id="private-another-heldout/query", cell_id="private-another-heldout")
    second["views"]["history"]["current.config.epochs"] = 200.0
    evidence[second["example_id"]] = copy.deepcopy(evidence[rows[1]["example_id"]])
    first = benchmark.prompt_payload(query, rows, labels, aliases, evidence, order)
    other = benchmark.prompt_payload(second, list(reversed(rows)), labels, aliases, evidence, order)
    assert first["context"] == other["context"]
    assert first["query"] != other["query"]
    text = benchmark.old.canonical(first)
    assert text.index('"context":') < text.index('"query":')
    other_text = benchmark.old.canonical(other)
    assert text.split(',"query":')[0] == other_text.split(',"query":')[0]


def test_exact_augmented_vectors_dataset_context_and_only_permitted_train_scores():
    rows, labels, query, aliases, evidence, _, order, payload = payload_fixture()
    assert set(payload) == {"context", "query"}
    assert set(payload["query"]) == {"features", "dataset_context"}
    assert payload["query"]["features"] == [query["views"]["history"][key] for key in order]
    assert len(payload["query"]["features"]) == 208
    assert payload["query"]["dataset_context"] == benchmark.compact_context(evidence[query["example_id"]]["context"])
    context = payload["context"]
    assert set(context) == {"task", "feature_order", "dataset_contexts", "training_pool"}
    assert context["feature_order"] == order and len(order) == 208
    assert len(context["dataset_contexts"]) == 2  # deduplicated by meaning, not by identity
    by_id = {row["example_id"]: row for row in rows}
    for item, (alias, info) in zip(context["training_pool"], aliases.items()):
        assert set(item) == {"alias", "session_alias", "features", "dataset_context_key", "official_accuracy", "official_delta"}
        row = by_id[info["example_id"]]
        assert item["alias"] == alias and item["session_alias"] == info["session_alias"]
        assert item["features"] == [row["views"]["history"][key] for key in order]
        assert context["dataset_contexts"][item["dataset_context_key"]] == benchmark.compact_context(evidence[row["example_id"]]["context"])
        assert item["official_accuracy"] == labels[row["example_id"]]["accuracy"]
        assert item["official_delta"] == labels[row["example_id"]]["delta_accuracy"]
    text = benchmark.old.canonical(payload)
    assert "private-" not in text
    assert "do-not-send" not in text
    assert "0.987654321234" not in text
    assert '"provenance"' not in text
    assert '"code"' not in text


@pytest.mark.parametrize("invalid", [
    "heldout_row", "session_overlap", "benchmark", "extra_test_label", "missing_target",
    "missing_parent", "missing_delta", "duplicate_id", "bad_alias", "reference_feature",
])
def test_prompt_train_only_score_and_identity_boundaries(invalid):
    rows, labels, query, aliases, evidence, _, order = fixture()
    if invalid == "heldout_row":
        rows[0]["split"] = "test"
    elif invalid == "session_overlap":
        query["cell_id"] = rows[0]["cell_id"]
    elif invalid == "benchmark":
        query["benchmark"] = "aime2025"
    elif invalid == "extra_test_label":
        labels[query["example_id"]] = {"accuracy": 0.9, "delta_accuracy": 0.8}
    elif invalid == "missing_target":
        labels[rows[0]["example_id"]]["accuracy"] = None
    elif invalid == "missing_parent":
        rows[0]["parent_reference"] = None
    elif invalid == "missing_delta":
        labels[rows[0]["example_id"]].pop("delta_accuracy")
    elif invalid == "duplicate_id":
        rows[1] = copy.deepcopy(rows[0])
    elif invalid == "bad_alias":
        aliases[next(iter(aliases))]["example_id"] = "private-heldout/query"
    else:
        rows[0]["views"]["history"]["parent.accuracy"] = 0.9
    with pytest.raises(ValueError):
        benchmark.prompt_payload(query, rows, labels, aliases, evidence, order)


@pytest.mark.parametrize("invalid", ["nan", "boolean", "missing", "extra_matching_order", "duplicate_order"])
def test_vector_requires_exact_finite_208_feature_whitelist(invalid):
    rows, _, _, _, _, _, order = fixture()
    row = rows[0]
    if invalid == "nan":
        row["views"]["history"][order[0]] = float("nan")
    elif invalid == "boolean":
        row["views"]["history"][order[0]] = True
    elif invalid == "missing":
        row["views"]["history"].pop(order[0])
    elif invalid == "extra_matching_order":
        row["views"]["history"]["future_target_accuracy"] = 0.9
        order.append("future_target_accuracy")
    else:
        order.append(order[0])
    with pytest.raises(ValueError):
        benchmark.vector(row, order)


@pytest.mark.parametrize("target", ["accuracy", "delta"])
def test_resolve_retains_all_51_rows_default_one_and_maps_overrides_by_identity(target):
    rows, _, query, aliases, _, response, _ = fixture()
    response["target"] = target
    alias_order = list(aliases)
    response["weight_overrides"] = {alias_order[0]: 2.0, alias_order[-1]: 0.5}
    before = copy.deepcopy((rows, query, aliases, response))
    spec, ignored = benchmark.resolve_spec(response, aliases, rows, query)
    assert ignored == {}
    assert set(spec) == {"weights", "target", "model", "feature_keys", "rationale"}
    assert len(spec["weights"]) == 51
    alias_of = {info["example_id"]: alias for alias, info in aliases.items()}
    assert spec["weights"] == [response["weight_overrides"].get(alias_of[row["example_id"]], 1.0) for row in rows]
    assert spec["weights"].count(1.0) == 49
    assert spec["target"] == target
    assert before == (rows, query, aliases, response)
    reversed_spec, _ = benchmark.resolve_spec(response, aliases, list(reversed(rows)), query)
    assert reversed_spec["weights"] == list(reversed(spec["weights"]))


def test_extra_metadata_is_projected_and_returned_without_affecting_spec():
    rows, _, query, aliases, _, response, _ = fixture()
    original, _ = benchmark.resolve_spec(response, aliases, rows, query)
    extras = {"weights_note": "Broad evidence.", "cost_usd": 0.1, "additional_notes": {"unknown": True}}
    response.update(extras)
    resolved, ignored = benchmark.resolve_spec(response, aliases, rows, query)
    assert resolved == original
    assert ignored == extras


@pytest.mark.parametrize("invalid", [
    "only_subset", "missing_target", "invalid_target", "invalid_model", "missing_parent_feature",
    "unknown_feature", "too_many_features", "unknown_alias", "raw_id", "override_not_object",
    "short_pool", "duplicate_feature",
])
def test_invalid_broad_schema_subset_attempts_and_overrides_rejected(invalid):
    rows, _, query, aliases, _, response, _ = fixture()
    if invalid == "only_subset":
        response = {"selected_aliases": list(aliases)[:8], "model": "ridge_10", "target": "delta"}
    elif invalid == "missing_target":
        response.pop("target")
    elif invalid == "invalid_target":
        response["target"] = "positive_delta_only"
    elif invalid == "invalid_model":
        response["model"] = "unrestricted_code"
    elif invalid == "missing_parent_feature":
        response["feature_keys"] = ["recipe_data.current.family.gsm8k"]
    elif invalid == "unknown_feature":
        response["feature_keys"].append("actual_query_accuracy")
    elif invalid == "too_many_features":
        response["feature_keys"] = list(FEATURE_KEYS[:13])
    elif invalid == "unknown_alias":
        response["weight_overrides"] = {"T999": 1.0}
    elif invalid == "raw_id":
        response["weight_overrides"] = {rows[0]["example_id"]: 1.0}
    elif invalid == "override_not_object":
        response["weight_overrides"] = [1.0] * 51
    elif invalid == "short_pool":
        rows = rows[:-1]
        aliases = benchmark.old.aliases_for(rows)
    else:
        response["feature_keys"].append("parent.accuracy")
    with pytest.raises(ValueError):
        benchmark.resolve_spec(response, aliases, rows, query)


@pytest.mark.parametrize("bad", [None, True, "1", 0, -1, 0.49, 2.01, float("nan"), float("inf")])
def test_override_weights_cannot_drop_examples_or_exceed_bounds(bad):
    rows, _, query, aliases, _, response, _ = fixture()
    response["weight_overrides"] = {next(iter(aliases)): bad}
    with pytest.raises(ValueError):
        benchmark.resolve_spec(response, aliases, rows, query)


def test_direct_prompt_weight_alignment_and_no_selector_decision_or_prediction():
    rows, _, _, aliases, _, response, _, payload = payload_fixture()
    weights = [0.7 + i / 100 for i in range(len(rows))]
    text = benchmark.direct_prompt(payload, rows, aliases, weights)
    direct = json.loads(text)
    assert set(direct) == {"context", "query", "weighting"}
    assert direct["context"] == payload["context"]
    assert direct["query"] == payload["query"]
    assert set(direct["weighting"]) == {"effective_weights"}
    weight_by_id = {row["example_id"]: weight for row, weight in zip(rows, weights)}
    expected = [weight_by_id[info["example_id"]] for info in aliases.values()]
    assert direct["weighting"]["effective_weights"] == expected
    assert [row["alias"] for row in direct["context"]["training_pool"]] == list(aliases)
    assert benchmark.direct_prompt(payload, list(reversed(rows)), aliases, list(reversed(weights))) == text
    assert text.index('"context":') < text.index('"query":') < text.index('"weighting":')
    assert response["rationale"] not in text
    assert response["model"] not in text
    for forbidden in ('"spec"', '"rationale"', '"target"', '"model"', '"predicted_accuracy"',
                      '"raw_prediction"', '"coefficients"', '"selected_ids"'):
        assert forbidden not in text
    assert "private-" not in text


@pytest.mark.parametrize("bad", [None, True, "1", 0, -1, float("nan"), float("inf")])
def test_direct_prompt_rejects_invalid_effective_weights(bad):
    rows, _, _, aliases, _, _, _, payload = payload_fixture()
    weights = [1.0] * len(rows)
    weights[0] = bad
    with pytest.raises(ValueError):
        benchmark.direct_prompt(payload, rows, aliases, weights)


@pytest.mark.parametrize("invalid", ["short_weights", "reordered_aliases", "wrong_payload_alias_order"])
def test_direct_prompt_rejects_weight_alias_misalignment(invalid):
    rows, _, _, aliases, _, _, _, payload = payload_fixture()
    weights = [1.0] * len(rows)
    if invalid == "short_weights":
        weights.pop()
    elif invalid == "reordered_aliases":
        aliases = dict(reversed(list(aliases.items())))
    else:
        payload["context"]["training_pool"].reverse()
    with pytest.raises(ValueError):
        benchmark.direct_prompt(payload, rows, aliases, weights)


def test_known_zero_training_accuracy_remains_in_context():
    _, _, _, _, _, _, _, payload = payload_fixture()
    assert any(row["official_accuracy"] == 0.0 for row in payload["context"]["training_pool"])
