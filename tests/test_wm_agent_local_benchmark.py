"""Synthetic-only prompt/selection boundaries; no fits, inference or data writes."""

import copy
import json
import re

import pytest

from tools.outcome_prediction import wm_agent_local_benchmark as benchmark
from tools.outcome_prediction.wm_agent_local_fit import FEATURE_KEYS


def fixture():
    pool, labels = [], {}
    for i in range(16):
        reference = 0.1 + i / 100
        values = dict.fromkeys(FEATURE_KEYS, 0.0)
        values.update({
            "parent.accuracy": reference, "parent.reference.measured": 1.0,
            "current.config.epochs": float(i),
        })
        row = {
            "example_id": f"private-training-session-{i // 2}/private-exp-{i}",
            "cell_id": f"private-training-session-{i // 2}", "benchmark": "gsm8k",
            "scientist_model": "private-scientist-identity", "split": "train",
            "parent_reference": reference, "reference_kind": "measured_parent",
            "views": {"history": values},
        }
        pool.append(row)
        target = reference + i / 100
        labels[row["example_id"]] = {"accuracy": target, "delta_accuracy": target - reference}
    query = copy.deepcopy(pool[0])
    query.update(example_id="private-heldout/query", cell_id="private-heldout", split="test")
    query["views"]["history"]["current.config.epochs"] = 100.0
    aliases = benchmark.aliases_for(pool)
    response = {
        "selected_aliases": list(aliases)[:8], "weights": [1.0] * 8,
        "model": "ridge_10", "feature_keys": ["parent.accuracy", "current.config.epochs"],
        "rationale": "Private selector rationale that must not reach direct inference.",
    }
    return pool, labels, query, aliases, response, list(FEATURE_KEYS)


def test_aliases_are_deterministic_opaque_and_session_consistent():
    pool, _, _, aliases, _, _ = fixture()
    assert aliases == benchmark.aliases_for(list(reversed(pool)))
    assert all(re.fullmatch(r"T\d{3}", alias) for alias in aliases)
    assert all(re.fullmatch(r"S\d{3}", info["session_alias"]) for info in aliases.values())
    assert {info["example_id"] for info in aliases.values()} == {row["example_id"] for row in pool}
    by_id = {row["example_id"]: row for row in pool}
    by_session = {}
    for info in aliases.values():
        session = by_id[info["example_id"]]["cell_id"]
        by_session.setdefault(session, set()).add(info["session_alias"])
    assert len(by_session) == 8
    assert all(len(values) == 1 for values in by_session.values())
    assert len({next(iter(values)) for values in by_session.values()}) == 8


@pytest.mark.parametrize("invalid", ["empty", "heldout", "duplicate", "mixed_benchmark"])
def test_alias_pool_boundaries(invalid):
    pool, *_ = fixture()
    if invalid == "empty":
        pool = []
    elif invalid == "heldout":
        pool[0]["split"] = "test"
    elif invalid == "duplicate":
        pool[1] = copy.deepcopy(pool[0])
    else:
        pool[0]["benchmark"] = "aime2025"
    with pytest.raises(ValueError):
        benchmark.aliases_for(pool)


def test_selector_prompt_exposes_only_train_outcomes_and_opaque_handles():
    pool, labels, query, aliases, _, order = fixture()
    text = benchmark.selector_prompt(query, pool, labels, aliases, order)
    payload = json.loads(text)
    assert set(payload) == {"task", "feature_order", "training_pool", "query_features"}
    assert payload["feature_order"] == order
    assert payload["query_features"] == [query["views"]["history"][key] for key in order]
    assert len(payload["training_pool"]) == len(pool)
    by_id = {row["example_id"]: row for row in pool}
    for item, (alias, info) in zip(payload["training_pool"], aliases.items()):
        assert set(item) == {"alias", "session_alias", "features", "official_accuracy", "official_delta"}
        assert item["alias"] == alias
        assert item["session_alias"] == info["session_alias"]
        row = by_id[info["example_id"]]
        assert item["features"] == [row["views"]["history"][key] for key in order]
        assert item["official_accuracy"] == labels[row["example_id"]]["accuracy"]
        assert item["official_delta"] == labels[row["example_id"]]["delta_accuracy"]
    assert "private-" not in text
    assert "query_accuracy" not in text


@pytest.mark.parametrize("invalid", ["heldout_pool", "session_overlap", "benchmark", "extra_test_label"])
def test_selector_prompt_training_boundaries(invalid):
    pool, labels, query, aliases, _, order = fixture()
    if invalid == "heldout_pool":
        pool[0]["split"] = "test"
    elif invalid == "session_overlap":
        query["cell_id"] = pool[0]["cell_id"]
    elif invalid == "benchmark":
        query["benchmark"] = "aime2025"
    else:
        labels[query["example_id"]] = {"accuracy": 0.99, "delta_accuracy": 0.89}
    with pytest.raises(ValueError):
        benchmark.selector_prompt(query, pool, labels, aliases, order)


def test_resolve_selection_exact_alias_mapping_preserves_order_and_weights():
    pool, _, query, aliases, response, _ = fixture()
    response["selected_aliases"].reverse()
    response["weights"] = [0.25, 0.5, 1, 2, 4, 1, 0.75, 1.5]
    before = copy.deepcopy((pool, query, aliases, response))
    result = benchmark.resolve_selection(response, aliases, pool, query)
    assert set(result) == {"selected_ids", "weights", "model", "feature_keys", "rationale"}
    assert result["selected_ids"] == [aliases[alias]["example_id"] for alias in response["selected_aliases"]]
    assert result["weights"] == response["weights"]
    assert result["model"] == response["model"]
    assert result["rationale"] == response["rationale"]
    assert before == (pool, query, aliases, response)


@pytest.mark.parametrize("invalid", [
    "missing_key", "extra_key", "unknown_alias", "duplicate_alias", "alias_not_list",
    "alias_not_string", "raw_id_not_alias", "bad_model", "bad_feature", "missing_parent",
])
def test_resolve_selection_rejects_malformed_schema_or_selection(invalid):
    pool, _, query, aliases, response, _ = fixture()
    if invalid == "missing_key":
        del response["rationale"]
    elif invalid == "extra_key":
        response["predicted_accuracy"] = 0.9
    elif invalid == "unknown_alias":
        response["selected_aliases"][0] = "T999"
    elif invalid == "duplicate_alias":
        response["selected_aliases"][0] = response["selected_aliases"][1]
    elif invalid == "alias_not_list":
        response["selected_aliases"] = "T001"
    elif invalid == "alias_not_string":
        response["selected_aliases"][0] = 1
    elif invalid == "raw_id_not_alias":
        response["selected_aliases"][0] = pool[0]["example_id"]
    elif invalid == "bad_model":
        response["model"] = "execute_arbitrary_code"
    elif invalid == "bad_feature":
        response["feature_keys"].append("target_accuracy")
    else:
        response["feature_keys"] = ["current.config.epochs"]
    with pytest.raises(ValueError):
        benchmark.resolve_selection(response, aliases, pool, query)


@pytest.mark.parametrize("bad", [None, True, "1", 0, -1, 0.24, 4.01, float("nan"), float("inf")])
def test_resolve_selection_rejects_invalid_raw_weights(bad):
    pool, _, query, aliases, response, _ = fixture()
    response["weights"][0] = bad
    with pytest.raises(ValueError):
        benchmark.resolve_selection(response, aliases, pool, query)


def test_selected_llm_prompt_has_exact_selected_data_weights_and_no_selector_context():
    pool, labels, query, aliases, response, order = fixture()
    selection = benchmark.resolve_selection(response, aliases, pool, query)
    by_id = {row["example_id"]: row for row in pool}
    selected = [by_id[key] for key in selection["selected_ids"]]
    selected_labels = {row["example_id"]: labels[row["example_id"]] for row in selected}
    weights = [0.5, 1.5] * 4
    text = benchmark.selected_prompt(query, selected, selected_labels, weights, order)
    payload = json.loads(text)
    assert set(payload) == {"task", "feature_order", "labeled_training_demonstrations", "target_features"}
    demonstrations = payload["labeled_training_demonstrations"]
    assert len(demonstrations) == 8
    for item, row, weight in zip(demonstrations, selected, weights):
        assert set(item) == {"features", "official_accuracy", "official_delta", "effective_fit_weight"}
        assert item["features"] == [row["views"]["history"][key] for key in order]
        assert item["official_accuracy"] == selected_labels[row["example_id"]]["accuracy"]
        assert item["official_delta"] == selected_labels[row["example_id"]]["delta_accuracy"]
        assert item["effective_fit_weight"] == weight
    assert payload["target_features"] == [query["views"]["history"][key] for key in order]
    assert "private-" not in text
    assert response["rationale"] not in text
    assert response["model"] not in text
    assert '"rationale"' not in text
    assert '"model"' not in text
    assert '"predicted_accuracy"' not in text
    assert '"selected_aliases"' not in text
    assert '"session_alias"' not in text
    assert all(alias not in text for alias in aliases)


@pytest.mark.parametrize("invalid", ["heldout_row", "session_overlap", "benchmark", "extra_test_label", "extra_unselected_label"])
def test_selected_prompt_rejects_unsafe_demonstrations(invalid):
    pool, labels, query, _, _, order = fixture()
    rows = pool[:8]
    subset_labels = {row["example_id"]: labels[row["example_id"]] for row in rows}
    if invalid == "heldout_row":
        rows[0]["split"] = "test"
    elif invalid == "session_overlap":
        query["cell_id"] = rows[0]["cell_id"]
    elif invalid == "benchmark":
        rows[0]["benchmark"] = "aime2025"
    elif invalid == "extra_test_label":
        subset_labels[query["example_id"]] = {"accuracy": 0.99, "delta_accuracy": 0.89}
    else:
        subset_labels[pool[8]["example_id"]] = labels[pool[8]["example_id"]]
    with pytest.raises(ValueError):
        benchmark.selected_prompt(query, rows, subset_labels, [1.0] * len(rows), order)


@pytest.mark.parametrize("bad", [None, True, "1", 0, -1, float("nan"), float("inf")])
def test_selected_prompt_rejects_invalid_effective_weights(bad):
    pool, labels, query, _, _, order = fixture()
    weights = [1.0] * len(pool)
    weights[0] = bad
    with pytest.raises(ValueError):
        benchmark.selected_prompt(query, pool, labels, weights, order)


def test_selected_prompt_rejects_wrong_weight_length():
    pool, labels, query, _, _, order = fixture()
    with pytest.raises(ValueError, match="weights"):
        benchmark.selected_prompt(query, pool, labels, [1.0], order)


@pytest.mark.parametrize("function", ["selector", "selected"])
@pytest.mark.parametrize("invalid", ["missing_target", "missing_parent", "missing_delta", "nonfinite_feature", "extra_feature"])
def test_both_prompt_types_reject_dirty_or_nonwhitelisted_data(function, invalid):
    pool, labels, query, aliases, _, order = fixture()
    if invalid == "missing_target":
        labels[pool[0]["example_id"]]["accuracy"] = None
    elif invalid == "missing_parent":
        pool[0]["parent_reference"] = None
    elif invalid == "missing_delta":
        labels[pool[0]["example_id"]].pop("delta_accuracy")
    elif invalid == "nonfinite_feature":
        pool[0]["views"]["history"]["current.config.epochs"] = float("nan")
    else:
        pool[0]["views"]["history"]["target_accuracy"] = 0.9
    with pytest.raises(ValueError):
        if function == "selector":
            benchmark.selector_prompt(query, pool, labels, aliases, order)
        else:
            benchmark.selected_prompt(query, pool, labels, [1.0] * len(pool), order)


def test_real_zero_training_accuracy_is_exposed_not_discarded():
    pool, labels, query, _, _, order = fixture()
    labels[pool[0]["example_id"]] = {"accuracy": 0.0, "delta_accuracy": -pool[0]["parent_reference"]}
    payload = json.loads(benchmark.selected_prompt(query, pool, labels, [1.0] * len(pool), order))
    assert payload["labeled_training_demonstrations"][0]["official_accuracy"] == 0.0
