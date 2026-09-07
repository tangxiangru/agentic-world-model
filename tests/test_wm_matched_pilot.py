"""Synthetic matched-pilot contracts and mocked orchestration; no real data/calls."""

import copy
import json
from types import SimpleNamespace

import pytest

from tools.outcome_prediction import wm_matched_fit as fit
from tools.outcome_prediction import wm_matched_pilot as pilot
from tools.outcome_prediction.wm_agent_broad_fit import FEATURE_KEYS
from tools.outcome_prediction.wm_agent_recipe_data import extract as data_extract
from tools.outcome_prediction.wm_matched_evidence import extract as treatment_extract


def fixture(n=8):
    source = {"model_input": {"plan": {"setup": {"method": {"family": "sft"},
                                                "data": [{"source": "openai/gsm8k", "selection": "keep only correct solutions"}]}}, "code": []},
              "history": [], "history_complete_to_base": True}
    dataset, treatment = data_extract(source), treatment_extract(source)
    dataset["provenance"]["private"] = "PRIVATE_PROVENANCE_CANARY"
    treatment["provenance"]["private"] = "PRIVATE_PROVENANCE_CANARY"
    train, labels, datasets, treatments = [], {}, {}, {}
    for i in range(n):
        parent = 0.5 + i / 1000
        values = dict.fromkeys(FEATURE_KEYS, 0.0)
        values.update(dataset["features"])
        values.update({"parent.accuracy": parent, "parent.reference.measured": 1.0,
                       "current.family.sft": 1.0, "current.config.epochs": 1.0,
                       "current.observed.config.epochs": 1.0})
        identity = f"private-session-{i // 2}/private-exp-{i}"
        train.append({"example_id": identity, "cell_id": f"private-session-{i // 2}", "benchmark": "gsm8k",
                      "scientist_model": "PRIVATE_SCIENTIST", "split": "train", "parent_reference": parent,
                      "reference_kind": "measured_parent", "views": {"history": values}})
        target = 0.65 + i / 1000
        labels[identity] = {"accuracy": target, "delta_accuracy": target - parent}
        datasets[identity], treatments[identity] = copy.deepcopy(dataset), copy.deepcopy(treatment)
    query = copy.deepcopy(train[0])
    query.update(example_id="private-heldout/query", cell_id="private-heldout", split="test")
    datasets[query["example_id"]], treatments[query["example_id"]] = copy.deepcopy(dataset), copy.deepcopy(treatment)
    return train, labels, query, datasets, treatments, list(FEATURE_KEYS)


def policy():
    return {"feature_order": list(FEATURE_KEYS), "max_attempts": 2, "workers": 1,
            "selector_system": pilot.MATCH_SYSTEM, "predictor_system": pilot.DIRECT_SYSTEM,
            "reported_cost_stop_usd": 10.0, "per_call_budget_usd": 0.5,
            "max_output_tokens": 3072, "timeout_seconds": 240, "effort": "high", "model_requested": "claude-opus-5",
            "limitations": []}


def test_matching_packet_is_outcome_blind_exact208_and_context_only():
    train, _, query, datasets, treatments, order = fixture()
    prompt, aliases = pilot.packet(query, train, datasets, treatments, order)
    payload = json.loads(prompt)
    assert payload["feature_order"] == order and len(order) == 208
    assert set(payload["query"]) == {"features", "context"}
    assert payload["query"]["features"] == [query["views"]["history"][key] for key in order]
    assert len(payload["examples"]) == len(train)
    for example, (alias, info) in zip(payload["examples"], aliases.items()):
        assert set(example) == {"alias", "session_alias", "features", "context"}
        assert example["alias"] == alias and example["session_alias"] == info["session_alias"]
        assert set(example["context"]) == {"dataset", "treatment"}
        assert len(example["features"]) == 208
    assert all(token not in prompt for token in ("official_accuracy", "official_delta", "provenance", "private-", "PRIVATE_"))


def test_empty_candidate_packet_and_empty_match_are_valid_unsupported_evidence():
    train, _, query, datasets, treatments, order = fixture()
    prompt, aliases = pilot.packet(query, [], datasets, treatments, order)
    assert aliases == {} and json.loads(prompt)["examples"] == []
    selected, ignored = pilot.resolve({"selected_aliases": [], "weights": [], "rationale": "No comparable candidates."},
                                      aliases, train, query, treatments)
    assert not selected["neighborhood"]["support"]["supported"]
    assert selected["selected_ids"] == [] and ignored == {}


def test_direct_packet_same_inputs_labels_and_weights_follow_identity_not_alias_sort():
    train, labels, query, datasets, treatments, order = fixture()
    train.reverse()
    weights = [0.5 + index / 10 for index in range(len(train))]
    plain, aliases = pilot.packet(query, train, datasets, treatments, order)
    direct, other_aliases = pilot.packet(query, train, datasets, treatments, order, labels=labels,
                                        weights=weights, kind="agent_matched_training")
    plain, direct = json.loads(plain), json.loads(direct)
    assert aliases == other_aliases and direct["query"] == plain["query"]
    weight_of = dict(zip([r["example_id"] for r in train], weights))
    for original, enriched, (_, info) in zip(plain["examples"], direct["examples"], aliases.items()):
        assert {k: v for k, v in enriched.items() if k not in {"official_accuracy", "official_delta", "effective_weight"}} == original
        identity = info["example_id"]
        assert enriched["official_accuracy"] == labels[identity]["accuracy"]
        assert enriched["official_delta"] == labels[identity]["delta_accuracy"]
        assert enriched["effective_weight"] == weight_of[identity]
    assert "official_accuracy" not in direct["query"]


@pytest.mark.parametrize("case", [
    "duplicate", "test_row", "query_session", "benchmark", "extra_label", "missing_label", "none_label",
    "missing_target", "nan_target", "bool_target", "range_target", "wrong_delta", "none_parent",
    "missing_weights", "weight_length", "weight_zero", "weight_nan", "weight_bool", "extra_feature", "bad_evidence",
])
def test_packet_rejects_invalid_pool_labels_weights_and_metadata(case):
    train, labels, query, datasets, treatments, order = fixture()
    key = train[0]["example_id"]
    weights = [1.0] * len(train)
    if case == "duplicate":
        train[1] = copy.deepcopy(train[0])
    elif case == "test_row":
        train[0]["split"] = "test"
    elif case == "query_session":
        query["cell_id"] = train[0]["cell_id"]
    elif case == "benchmark":
        train[0]["benchmark"] = "aime2025"
    elif case == "extra_label":
        labels[query["example_id"]] = {"accuracy": 0.9, "delta_accuracy": 0.4}
    elif case == "missing_label":
        labels.pop(key)
    elif case == "none_label":
        labels[key] = None
    elif case == "missing_target":
        labels[key].pop("accuracy")
    elif case == "nan_target":
        labels[key]["accuracy"] = float("nan")
    elif case == "bool_target":
        labels[key]["accuracy"] = True
    elif case == "range_target":
        labels[key]["accuracy"] = 2.0
    elif case == "wrong_delta":
        labels[key]["delta_accuracy"] += 0.01
    elif case == "none_parent":
        train[0]["parent_reference"] = None
    elif case == "missing_weights":
        weights = None
    elif case == "weight_length":
        weights.pop()
    elif case == "weight_zero":
        weights[0] = 0.0
    elif case == "weight_nan":
        weights[0] = float("nan")
    elif case == "weight_bool":
        weights[0] = True
    elif case == "extra_feature":
        query["views"]["history"]["target_accuracy"] = 0.9
    elif case == "bad_evidence":
        treatments[key]["context"]["current"]["evidence"]["target"] = 0.9
    with pytest.raises((ValueError, TypeError)):
        pilot.packet(query, train, datasets, treatments, order, labels=labels, weights=weights, kind="direct")


def test_resolve_uses_exact_aliases_retains_extra_metadata_and_is_label_free():
    train, _, query, datasets, treatments, order = fixture()
    _, aliases = pilot.packet(query, train, datasets, treatments, order)
    names = list(aliases)[1:7]
    response = {"selected_aliases": names, "weights": [1.0] * 6, "rationale": "Same input-defined treatment.",
                "harmless_note": "Preserve separately."}
    result, ignored = pilot.resolve(response, aliases, train, query, treatments)
    assert result["selected_ids"] == [aliases[name]["example_id"] for name in names]
    assert result["neighborhood"]["support"]["supported"]
    assert ignored == {"harmless_note": "Preserve separately."}
    assert "accuracy" not in response and "delta_accuracy" not in response


@pytest.mark.parametrize("case", ["unknown", "duplicate", "nonstring", "weights_length", "weight_bound", "empty_reason", "missing"])
def test_resolve_rejects_nonexact_selection_or_malformed_spec(case):
    train, _, query, datasets, treatments, order = fixture()
    _, aliases = pilot.packet(query, train, datasets, treatments, order)
    response = {"selected_aliases": list(aliases)[:6], "weights": [1.0] * 6, "rationale": "Safe match."}
    if case == "unknown":
        response["selected_aliases"][0] = "T999"
    elif case == "duplicate":
        response["selected_aliases"][1] = response["selected_aliases"][0]
    elif case == "nonstring":
        response["selected_aliases"][0] = 1
    elif case == "weights_length":
        response["weights"].pop()
    elif case == "weight_bound":
        response["weights"][0] = 0.0
    elif case == "empty_reason":
        response["rationale"] = "  "
    else:
        response.pop("weights")
    with pytest.raises((ValueError, TypeError)):
        pilot.resolve(response, aliases, train, query, treatments)


def test_numeric_forecasts_supply_only_pool_labels_never_query_target(monkeypatch):
    train, labels, query, _, treatments, _ = fixture()
    labels[query["example_id"]] = {"accuracy": 0.999123, "delta_accuracy": 0.499123}
    seen = []

    def spy(rows, supplied, target, variant, *, evidence):
        assert set(supplied) == {row["example_id"] for row in rows}
        assert target["example_id"] not in supplied
        assert evidence is treatments
        seen.append(variant)
        return {"predicted_accuracy": 0.5}

    monkeypatch.setattr(fit, "fit_predict", spy)
    result = pilot.numeric_forecasts(train, labels, [query], treatments)
    assert set(result[0]["fits"]) == set(pilot.VARIANTS)
    assert seen == list(pilot.VARIANTS)


def test_loso_removes_entire_training_session_before_matching_and_label_packet(monkeypatch):
    train, labels, _, _, treatments, _ = fixture()
    seen = []

    def spy(fitting, supplied, held, evidence):
        fit_sessions = {row["cell_id"] for row in fitting}
        held_sessions = {row["cell_id"] for row in held}
        assert len(held_sessions) == 1 and not fit_sessions & held_sessions
        assert len(held) == 2 and len(fitting) == len(train) - 2
        assert set(supplied) == {row["example_id"] for row in fitting}
        assert not {row["example_id"] for row in held} & set(supplied)
        seen.append(next(iter(held_sessions)))
        return [{"example_id": row["example_id"], "fits": {}} for row in held]

    monkeypatch.setattr(pilot, "numeric_forecasts", spy)
    result = pilot.loso_forecasts(train, labels, treatments)
    assert len(result) == len(train) and len(set(seen)) == 4
    assert [row["example_id"] for row in result] == sorted(row["example_id"] for row in train)


def run_mock(tmp_path, monkeypatch, mode):
    train, labels, query, datasets, treatments, order = fixture()
    config = policy()
    config["pilot_ids"] = [query["example_id"]]
    selection_prompt, aliases = pilot.packet(query, [] if mode == "empty_candidates" else train, datasets, treatments, order)
    fixed, _ = pilot.packet(query, train, datasets, treatments, order, labels=labels,
                            weights=pilot.uniform_session_weights(train), kind="fixed_16_session_diverse")
    packets = [{"example_id": query["example_id"], "selector": selection_prompt, "fixed16": fixed, "aliases": aliases}]
    bundle, output = tmp_path / "bundle", tmp_path / "results"
    bundle.mkdir()
    pilot.write(bundle / "manifest.json", {})
    monkeypatch.setattr(pilot, "validate_bundle", lambda _: (config, train, labels, [query], datasets, treatments, packets))
    monkeypatch.setattr(pilot, "numeric_forecasts", lambda *args: [])
    monkeypatch.setattr(pilot, "loso_forecasts", lambda *args: [])
    monkeypatch.setattr(pilot.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="mock-cli"))
    events, selector_calls, matched_packets = [], [], []
    real_packet = pilot.packet

    def checked_packet(*args, **kwargs):
        assert "selection_accepted" in events
        assert kwargs["kind"] == "agent_matched_training"
        assert set(kwargs["labels"]) == {row["example_id"] for row in args[1]}
        assert query["example_id"] not in kwargs["labels"]
        matched_packets.append(kwargs)
        events.append("labeled_matched_packet")
        return real_packet(*args, **kwargs)

    monkeypatch.setattr(pilot, "packet", checked_packet)
    original_resolve = pilot.resolve

    def checked_resolve(*args, **kwargs):
        result = original_resolve(*args, **kwargs)
        events.append("selection_accepted")
        return result

    monkeypatch.setattr(pilot, "resolve", checked_resolve)

    def invoke(prompt, invocation, budget):
        if prompt.startswith("Return exactly"):
            return {"attempted": True, "response": {"ok": mode != "probe_fail"},
                    "error": "mock_probe_failure" if mode == "probe_fail" else None}
        payload = json.loads(prompt)
        if invocation["system"] == pilot.MATCH_SYSTEM:
            assert all("official_accuracy" not in row and "official_delta" not in row for row in payload["examples"])
            selector_calls.append(prompt)
            events.append("selector_call")
            if mode == "selector_fail":
                return {"attempted": True, "response": None, "error": "mock_invocation_failure"}
            names = [] if mode == "unsupported" else list(aliases)[:6]
            if mode == "invalid_then_valid" and len(selector_calls) == 1:
                names = ["T999"]
            response = {"selected_aliases": names, "weights": [1.0] * len(names), "rationale": "Outcome-blind match.",
                        "harmless_extra": "saved"}
        else:
            events.append(payload["example_kind"])
            if payload["example_kind"] == "agent_matched_training":
                assert "selection_accepted" in events
                assert len(payload["examples"]) == 6
                if mode == "matched_fail":
                    return {"attempted": True, "response": None, "error": "mock_matched_failure"}
            response = {"predicted_accuracy": 0.56, "rationale": "Synthetic forecast."}
        return {"attempted": True, "response": response, "error": None}

    monkeypatch.setattr(pilot.recovery, "_invoke", invoke)
    monkeypatch.setattr(pilot, "historical_forecasts", lambda *args: pytest.fail("Historical outcomes accessed before freeze"))
    monkeypatch.setattr(pilot, "score", lambda *args: pytest.fail("Run must not score or read test outcomes"))
    result = pilot.run(bundle, output)
    frozen = pilot.read(output / "prediction_freeze.json")
    assert "agent_predictions.json" in frozen["files"]
    assert "numeric_predictions.json" in frozen["files"] and "train_loso_predictions.json" in frozen["files"]
    assert not (output / "report.json").exists()
    pilot.old.previous.verify_files(output, frozen["files"])
    record = pilot.read(output / "agent_predictions.json")[0]
    return result, record, events, selector_calls, matched_packets, output


def test_mocked_run_matches_before_training_labels_and_stops_at_first_valid(tmp_path, monkeypatch):
    _, record, events, calls, packets, output = run_mock(tmp_path, monkeypatch, "success")
    assert len(calls) == 1 and len(packets) == 1
    assert events.index("selection_accepted") < events.index("labeled_matched_packet")
    assert record["matched_llm"]["predicted_accuracy"] == record["fixed16_llm"]["predicted_accuracy"] == 0.56
    assert record["errors"] == {}
    saved = next((output / "calls").glob("*__selector__attempt1.json"))
    assert pilot.read(saved)["ignored_fields"] == {"harmless_extra": "saved"}


def test_mocked_run_first_semantically_valid_after_invalid_response(tmp_path, monkeypatch):
    _, record, _, calls, packets, output = run_mock(tmp_path, monkeypatch, "invalid_then_valid")
    assert len(calls) == 2 and calls[0] == calls[1]
    assert record["selector_attempts"] == 2 and len(packets) == 1
    failed = pilot.read(next((output / "calls").glob("*__selector__attempt1.json")))
    assert failed["error"].startswith("invalid_response:")


def test_valid_unsupported_selection_falls_back_but_still_calls_fixed_bank(tmp_path, monkeypatch):
    _, record, events, _, packets, _ = run_mock(tmp_path, monkeypatch, "unsupported")
    assert record["selection"] is not None and record["selection"]["selected_ids"] == []
    assert record["agent_median"]["predicted_accuracy"] == 0.5
    assert record["agent_shrunk"]["predicted_accuracy"] == 0.5
    assert record["matched_llm"]["fallback"] == "insufficient_matching_support"
    assert record["matched_attempts"] == 0 and not packets
    assert record["fixed16_llm"]["predicted_accuracy"] == 0.56
    assert "fixed_16_session_diverse" in events


def test_empty_candidate_run_preserves_explicit_parent_fallback(tmp_path, monkeypatch):
    _, record, _, calls, packets, _ = run_mock(tmp_path, monkeypatch, "empty_candidates")
    assert len(calls) == 1 and json.loads(calls[0])["examples"] == []
    assert not packets and record["selection"]["selected_ids"] == []
    assert record["matched_llm"]["predicted_accuracy"] == 0.5
    assert record["fixed16_llm"]["predicted_accuracy"] == 0.56


def test_failed_selector_is_null_not_fallback_and_fixed_bank_is_independent(tmp_path, monkeypatch):
    _, record, events, calls, packets, _ = run_mock(tmp_path, monkeypatch, "selector_fail")
    assert len(calls) == 2 and not packets
    assert all(record[key] is None for key in ("selection", "agent_median", "agent_shrunk", "matched_llm"))
    assert record["fixed16_llm"]["predicted_accuracy"] == 0.56
    assert "fixed_16_session_diverse" in events


def test_failed_matched_forecast_is_null_not_parent_fallback(tmp_path, monkeypatch):
    _, record, _, _, _, _ = run_mock(tmp_path, monkeypatch, "matched_fail")
    assert record["selection"]["neighborhood"]["support"]["supported"]
    assert record["agent_median"] is not None and record["matched_llm"] is None
    assert record["matched_attempts"] == 2 and record["fixed16_llm"] is not None


def test_probe_failure_preserves_numeric_outputs_and_null_agent_arms(tmp_path, monkeypatch):
    _, record, _, calls, packets, _ = run_mock(tmp_path, monkeypatch, "probe_fail")
    assert not calls and not packets
    assert record["selector_attempts"] == record["fixed16_attempts"] == 0
    assert all(record[key] is None for key in ("selection", "agent_median", "agent_shrunk", "matched_llm", "fixed16_llm"))


def raw_record(query, prompt, response, attempt=1):
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-5", "tools": [], "mcp_servers": [], "plugins": [], "skills": []},
        {"type": "assistant", "message": {"model": "claude-opus-5", "content": [{"type": "text", "text": "forecast"}]}},
        {"type": "result", "is_error": False, "subtype": "success", "result": json.dumps(response)},
    ]
    return {"example_id": query["example_id"], "stage": "fixed16", "attempt": attempt,
            "prompt_sha256": pilot.old.digest(prompt), "attempted": True, "max_output_tokens": 3072,
            "timeout_seconds": 240, "effort": "high", "stdout": "\n".join(json.dumps(event) for event in events), "stderr": "",
            "raw_events": events, "models_resolved": ["claude-opus-5"], "model_requested": "claude-opus-5",
            "returncode": 0, "response": response, "error": None, "ignored_fields": {}}


def verify_raw(tmp_path, query, records, expected):
    output = tmp_path / "raw_results"
    (output / "calls").mkdir(parents=True)
    for index, record in enumerate(records, 1):
        pilot.write(output / "calls" / (pilot.old.key_for(query) + f"__fixed16__attempt{index}.json"), record)
    pilot.verify_attempts(output, query, "fixed16", len(records), "PROMPT", policy(), lambda response: (response, {}), expected)


def test_raw_call_replay_success(tmp_path):
    _, _, query, _, _, _ = fixture()
    response = {"predicted_accuracy": 0.56, "rationale": "Synthetic."}
    verify_raw(tmp_path, query, [raw_record(query, "PROMPT", response)], response)


@pytest.mark.parametrize("field", ["prompt", "response", "events", "model", "allowance", "timeout", "effort", "error", "ignored", "expected"])
def test_raw_call_replay_rejects_tampering_before_scoring(tmp_path, field):
    _, _, query, _, _, _ = fixture()
    response = {"predicted_accuracy": 0.56, "rationale": "Synthetic."}
    record = raw_record(query, "PROMPT", response)
    expected = response
    if field == "prompt":
        record["prompt_sha256"] = "altered"
    elif field == "response":
        record["response"] = {"predicted_accuracy": 0.99}
    elif field == "events":
        record["raw_events"][0]["tools"] = ["read"]
    elif field == "model":
        record["model_requested"] = "different-model"
    elif field == "allowance":
        record["max_output_tokens"] += 1
    elif field == "timeout":
        record["timeout_seconds"] += 1
    elif field == "effort":
        record["effort"] = "low"
    elif field == "error":
        record["error"] = "invented"
    elif field == "ignored":
        record["ignored_fields"] = {"invented": 1}
    else:
        expected = {"predicted_accuracy": 0.99}
    with pytest.raises(ValueError):
        verify_raw(tmp_path, query, [record], expected)


def test_raw_call_replay_rejects_selection_after_an_earlier_valid_response(tmp_path):
    _, _, query, _, _, _ = fixture()
    first, second = {"value": 1}, {"value": 2}
    with pytest.raises(ValueError, match="first semantically valid"):
        verify_raw(tmp_path, query, [raw_record(query, "PROMPT", first), raw_record(query, "PROMPT", second, 2)], second)


@pytest.mark.parametrize("second_success", [False, True])
def test_execution_timeout_with_complete_stdout_remains_failed_not_accepted(tmp_path, second_success):
    _, _, query, _, _, _ = fixture()
    timed_out_value = {"predicted_accuracy": 0.99, "rationale": "Never accepted after timeout."}
    failed = raw_record(query, "PROMPT", timed_out_value)
    failed.update(response=None, error="request_timeout", returncode=None)
    records, expected = [failed], None
    if second_success:
        expected = {"predicted_accuracy": 0.56, "rationale": "Accepted fresh retry."}
        records.append(raw_record(query, "PROMPT", expected, 2))
    verify_raw(tmp_path, query, records, expected)


def test_score_requires_forecast_freeze_before_validation_or_target_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "validate_bundle", lambda *args: pytest.fail("Validated before prediction freeze"))
    with pytest.raises(FileNotFoundError):
        pilot.score(tmp_path / "bundle", tmp_path / "no-results")


def test_score_numeric_replay_must_pass_before_any_target_file_read(tmp_path, monkeypatch):
    train, labels, query, datasets, treatments, _ = fixture()
    bundle, output = tmp_path / "bundle", tmp_path / "results"
    bundle.mkdir()
    output.mkdir()
    pilot.write(bundle / "manifest.json", {})
    pilot.write(output / "prediction_freeze.json", {"files": {}})
    pilot.write(output / "run_policy.json", {"bundle_manifest_sha256": pilot.sha(bundle / "manifest.json")})
    for name in ("numeric_predictions.json", "train_loso_predictions.json", "agent_predictions.json"):
        pilot.write(output / name, [])
    monkeypatch.setattr(pilot, "validate_bundle", lambda *args: (policy(), train, labels, [query], datasets, treatments, []))
    monkeypatch.setattr(pilot, "numeric_forecasts", lambda *args: ["not-the-saved-forecast"])
    read = pilot.read

    def guarded_read(path):
        assert path.name != "labels.json", "Target file decoded before replay"
        return read(path)

    monkeypatch.setattr(pilot, "read", guarded_read)
    monkeypatch.setattr(pilot, "historical_forecasts", lambda *args: pytest.fail("History loaded before replay"))
    with pytest.raises(ValueError, match="replay failed"):
        pilot.score(bundle, output)
