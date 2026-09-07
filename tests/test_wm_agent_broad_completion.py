"""Synthetic higher-allowance completion; no real data/model calls or scoring."""

import copy
import json
from pathlib import Path

import pytest

from tools.outcome_prediction import wm_agent_broad_completion as completion
from tools.outcome_prediction.wm_agent_broad_fit import FEATURE_KEYS
from tools.outcome_prediction.wm_agent_recipe_data import extract
from tools.outcome_prediction.wm_clean_refresh import read, sha, write


def fixture():
    rows, labels, evidence = [], {}, {}
    dataset = extract({"model_input": {"plan": {"setup": {"data": [{"source": "openai/gsm8k"}]}}},
                       "history": [], "history_complete_to_base": True})
    for i in range(51):
        values = dict.fromkeys(FEATURE_KEYS, 0.0)
        values.update(dataset["features"])
        values.update({"parent.accuracy": 0.1, "parent.reference.measured": 1.0})
        identity = f"train-{i // 3}/exp-{i}"
        rows.append({"example_id": identity, "cell_id": f"train-{i // 3}", "benchmark": "gsm8k",
                     "split": "train", "parent_reference": 0.1, "reference_kind": "measured_parent",
                     "views": {"history": values}})
        labels[identity] = {"accuracy": i / 100, "delta_accuracy": i / 100 - 0.1}
        evidence[identity] = copy.deepcopy(dataset)
    queries = []
    for name in ("q-success", "q-missing-llm", "q-missing-spec"):
        query = copy.deepcopy(rows[0])
        query.update(example_id=name + "/exp-0", cell_id=name, split="test")
        queries.append(query)
        evidence[query["example_id"]] = copy.deepcopy(evidence[rows[0]["example_id"]])
    aliases = completion.broad.old.aliases_for(rows)
    response = {"weight_overrides": {}, "target": "accuracy", "model": "ridge_100",
                "feature_keys": ["parent.accuracy"], "rationale": "Synthetic fixed policy."}
    spec, _ = completion.broad.resolve_spec(response, aliases, rows, queries[0])
    policy = {"model_requested": "claude-opus-5", "effort": "high", "max_attempts": 2,
              "selector_system": completion.broad.SELECT_SYSTEM, "predictor_system": completion.broad.PREDICT_SYSTEM,
              "feature_order": list(FEATURE_KEYS), "max_cli_network_retries": 1, "timeout_seconds": 240,
              "selector_max_output_tokens": 4096, "predictor_max_output_tokens": 2048,
              "selector_budget_usd": 1.25, "predictor_budget_usd": 1.0}
    packets = []
    for query in queries:
        prompt = completion.broad.old.canonical(completion.broad.prompt_payload(
            query, rows, labels, aliases, evidence, list(FEATURE_KEYS)))
        packets.append({"example_id": query["example_id"], "prompt": prompt,
                        "prompt_sha256": completion.broad.old.digest(prompt)})
    return policy, rows, labels, queries, {"gsm8k": aliases}, evidence, packets, response, spec


def record_for(response, *, stage="selector", error=False, query=None, attempt=1, prompt_hash="hash"):
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-5",
         "tools": [], "mcp_servers": [], "plugins": [], "skills": []},
        {"type": "assistant", "message": {"model": "claude-opus-5",
                                             "content": [{"type": "text", "text": "synthetic"}]}},
        {"type": "result", "is_error": error,
         "subtype": "error_max_budget_usd" if error else "success",
         "result": json.dumps(response), "total_cost_usd": 0.8},
    ]
    record = {"stdout": "\n".join(json.dumps(event) for event in events), "stderr": "", "returncode": int(error),
              "raw_events": events, "response": None if error else response, "models_resolved": ["claude-opus-5"],
              "model_requested": "claude-opus-5", "effort": "high", "cost_usd_reported": 0.8,
              "error": "CLI request failed: error_max_budget_usd" if error else None,
              "example_id": query["example_id"] if query else "query", "stage": stage,
              "attempt": attempt, "prompt_sha256": prompt_hash, "max_output_tokens": 2048}
    if not error and stage == "selector":
        record["ignored_fields"] = {}
    return record


def setup_dirs(tmp_path):
    bundle, original, output = (tmp_path / name for name in ("bundle", "original", "completion"))
    bundle.mkdir()
    original.mkdir()
    write(bundle / "manifest.json", {})
    return bundle, original, output


def fake_fit(rows, labels, query, spec):
    assert set(labels) == {row["example_id"] for row in rows}
    assert query["example_id"] not in labels
    assert all(row["split"] == "train" and row["cell_id"] != query["cell_id"] for row in rows)
    return {"predicted_accuracy": 0.25, "fit_metadata": {"normalized_sample_weights": [1.0] * len(rows)},
            "synthetic_target": spec["target"], "synthetic_model": spec["model"]}


def build_original(bundle, source, data):
    _, rows, labels, queries, banks, evidence, packets, response, spec = data
    for name in ("calls", "direct_prompts", "query_results"):
        (source / name).mkdir()
    write(source / "run_policy.json", {"bundle_manifest_sha256": sha(bundle / "manifest.json"), "started_at": "synthetic"})
    write(source / "probe.json", {"synthetic_probe": "preserved"})
    results = []
    for i, query in enumerate(queries):
        identity, key = query["example_id"], completion.broad.old.key_for(query)
        value = {"example_id": identity, "benchmark": query["benchmark"],
                 "selector_prompt_sha256": packets[i]["prompt_sha256"], "spec": None,
                 "fits": None, "llm_prediction": None, "error": "CLI request failed: error_max_budget_usd"}
        count = 2 if i == 2 else 1
        for attempt in range(1, count + 1):
            record = record_for(response, query=query, attempt=attempt, prompt_hash=packets[i]["prompt_sha256"], error=i == 2)
            write(source / "calls" / (key + f"__selector__attempt{attempt}.json"), record)
        value["selector_attempts"] = count
        if i != 2:
            value["spec"] = copy.deepcopy(spec)
            value["fits"] = {"agent_target": fake_fit(rows, labels, query, spec),
                              "fixed_accuracy": fake_fit(rows, labels, query, {**spec, "target": "accuracy"}),
                              "fixed_delta": fake_fit(rows, labels, query, {**spec, "target": "delta"}),
                              "uniform_weights": fake_fit(rows, labels, query, spec)}
            payload = completion.broad.prompt_payload(query, rows, labels, banks["gsm8k"], evidence, list(FEATURE_KEYS))
            prompt = completion.broad.direct_prompt(payload, rows, banks["gsm8k"], [1.0] * len(rows))
            prompt_hash = completion.broad.old.digest(prompt)
            write(source / "direct_prompts" / (key + ".json"), {"example_id": identity, "prompt": prompt, "prompt_sha256": prompt_hash})
            forecast = {"predicted_accuracy": 0.3, "rationale": "Frozen original forecast."}
            count = 1 if i == 0 else 2
            for attempt in range(1, count + 1):
                record = record_for(forecast, stage="predictor", query=query, attempt=attempt, prompt_hash=prompt_hash, error=i != 0)
                write(source / "calls" / (key + f"__predictor__attempt{attempt}.json"), record)
            value["predictor_attempts"] = count
            if i == 0:
                value.update(llm_prediction=forecast, error=None)
        results.append(value)
        write(source / "query_results" / (key + ".json"), value)
    write(source / "predictions.json", results)
    write(source / "prediction_freeze.json", {"created_at": completion.broad.old.now(), "reported_cost_usd": 7.0,
                                              "files": {str(path.relative_to(source)): sha(path)
                                                        for path in source.rglob("*") if path.is_file()}})
    return results


def test_prepare_blinded_before_original_freeze_without_loading_any_scores(monkeypatch, tmp_path):
    bundle, source, output = setup_dirs(tmp_path)
    seen = []

    def tracked_read(path):
        seen.append(Path(path).name)
        assert "prediction" not in Path(path).name and "label" not in Path(path).name
        return read(path)

    monkeypatch.setattr(completion, "read", tracked_read)
    completion.prepare(bundle, source, output)
    assert seen == ["manifest.json"]
    policy = read(output / "completion_policy.json")
    assert policy["max_output_tokens"] == 4096
    assert policy["per_call_budget_usd"] == 2
    assert policy["additional_reported_cost_stop_usd"] == 40
    assert policy["extra_attempts_per_stage"] == 2
    assert "can change reasoning" in policy["allowance_sensitivity"]
    with pytest.raises(FileExistsError):
        completion.prepare(bundle, source, output)


def test_run_requires_original_freeze_and_unscored_state(tmp_path):
    bundle, source, output = setup_dirs(tmp_path)
    completion.prepare(bundle, source, output)
    with pytest.raises(FileNotFoundError):
        completion.run(output)
    write(source / "report.json", {"already": "scored"})
    with pytest.raises(ValueError, match="after original target scoring"):
        completion.run(output)


def test_prepare_rejects_already_scored_original(tmp_path):
    bundle, source, output = setup_dirs(tmp_path)
    write(source / "report.json", {"already": "scored"})
    with pytest.raises(ValueError, match="before original target scoring"):
        completion.prepare(bundle, source, output)


@pytest.mark.parametrize("invalid", ["stage", "query", "attempt", "prompt", "model", "tool", "partial_budget", "success_flag"])
def test_original_call_audit_no_partial_budget_recovery_or_alignment_relaxation(invalid):
    policy, rows, _, queries, banks, _, _, response, spec = fixture()
    query = queries[0]
    record = record_for(response, query=query)
    result = {"spec": spec, "llm_prediction": None}
    if invalid == "stage":
        record["stage"] = "predictor"
    elif invalid == "query":
        record["example_id"] = "different"
    elif invalid == "attempt":
        record["attempt"] = 2
    elif invalid == "prompt":
        record["prompt_sha256"] = "different"
    elif invalid in {"model", "tool"}:
        if invalid == "model":
            record["raw_events"][0]["model"] = "claude-other"
        else:
            record["raw_events"][0]["tools"] = ["Read"]
        record["stdout"] = "\n".join(json.dumps(event) for event in record["raw_events"])
    elif invalid == "partial_budget":
        record = record_for(response, query=query, error=True)
    else:
        record["error"] = "unexpected former rejection"
    with pytest.raises(ValueError):
        completion.audit_original_calls([record], "selector", result, query, rows, banks["gsm8k"], policy, "hash")


def test_failed_partial_budget_response_is_kept_failed_not_recovered():
    policy, rows, _, queries, banks, _, _, response, _ = fixture()
    query = queries[0]
    record = record_for(response, query=query, error=True)
    completion.audit_original_calls([record], "selector", {"spec": None, "llm_prediction": None},
                                    query, rows, banks["gsm8k"], policy, "hash")


def test_safe_snapshot_copy_is_byte_exact_and_never_overwrites(tmp_path):
    source, target = tmp_path / "source.json", tmp_path / "copy" / "source.json"
    write(source, {"preserve": [0, 1, "failure"]})
    completion.copy_frozen_file(source, target, sha(source))
    assert sha(source) == sha(target)
    with pytest.raises(FileExistsError):
        completion.copy_frozen_file(source, target, sha(source))
    with pytest.raises(ValueError):
        completion.copy_frozen_file(source, tmp_path / "wrong.json", "wrong-hash")


@pytest.mark.parametrize("fail_new", [False, True])
def test_full_completion_preserves_existing_stages_appends_calls_and_full_cost(monkeypatch, tmp_path, fail_new):
    data = fixture()
    policy, rows, _, queries, banks, _, _, response, _ = data
    bundle, source, output = setup_dirs(tmp_path)
    completion.prepare(bundle, source, output)
    originals = build_original(bundle, source, data)
    monkeypatch.setattr(completion.broad, "validate_bundle", lambda _: data[:7])
    fits, new_calls = [], []

    def controlled_fit(fit_rows, fit_labels, query, spec):
        fits.append(query["example_id"])
        return fake_fit(fit_rows, fit_labels, query, spec)

    def fake_call(prompt, call_policy):
        assert call_policy["model_requested"] == "claude-opus-5"
        assert call_policy["effort"] == "high"
        assert call_policy["max_output_tokens"] == 4096
        assert call_policy["per_call_budget_usd"] == 2.0
        stage = "selector" if call_policy["system"] == completion.broad.SELECT_SYSTEM else "predictor"
        if stage == "predictor":
            assert call_policy["system"] == completion.broad.PREDICT_SYSTEM
        new_calls.append((stage, prompt))
        reply = response if stage == "selector" else {"predicted_accuracy": 0.4, "rationale": "New synthetic forecast."}
        return record_for(reply, stage=stage, error=fail_new)

    def guarded_read(path):
        assert Path(path).name != "labels.json", "Held-out labels must never be opened"
        return read(path)

    monkeypatch.setattr(completion, "fit_predict", controlled_fit)
    monkeypatch.setattr(completion.recovery.io, "call_json", fake_call)
    monkeypatch.setattr(completion, "read", guarded_read)
    summary = completion.run(output)
    merged = {row["example_id"]: row for row in read(output / "predictions.json")}
    original_freeze = read(source / "prediction_freeze.json")
    for name, expected in original_freeze["files"].items():
        assert sha(output / "original_artifacts" / name) == expected
        assert sha(source / name) == expected
        if name.startswith(("calls/", "direct_prompts/")) or name == "run_policy.json":
            assert sha(output / name) == expected
    assert sha(output / "original_artifacts" / "prediction_freeze.json") == sha(source / "prediction_freeze.json")
    assert merged[queries[0]["example_id"]] == originals[0]
    assert merged[queries[1]["example_id"]]["spec"] == originals[1]["spec"]
    assert merged[queries[1]["example_id"]]["fits"] == originals[1]["fits"]
    assert read(output / "run_policy.json") == read(source / "run_policy.json")
    provenance = read(output / "completion_provenance.json")
    assert provenance[queries[0]["example_id"]]["mode"] == "preserved_original"
    assert provenance[queries[0]["example_id"]]["new_calls"] == []
    if fail_new:
        assert len(new_calls) == 4  # two extra selector + two extra predictor attempts
        assert not fits
        assert merged[queries[1]["example_id"]]["predictor_attempts"] == 4
        assert merged[queries[2]["example_id"]]["selector_attempts"] == 4
        assert merged[queries[2]["example_id"]]["spec"] is None
        assert summary["llm_success"] == 1
    else:
        assert len(new_calls) == 3  # one missing selector, two missing forecasts
        assert fits == [queries[2]["example_id"]] * 3
        assert merged[queries[1]["example_id"]]["predictor_attempts"] == 3
        assert merged[queries[2]["example_id"]]["selector_attempts"] == 3
        assert merged[queries[2]["example_id"]]["predictor_attempts"] == 1
        assert summary["llm_success"] == 3
        for query in queries:
            key = completion.broad.old.key_for(query)
            value = merged[query["example_id"]]
            for stage in ("selector", "predictor"):
                count = value[stage + "_attempts"]
                records = [read(output / "calls" / (key + f"__{stage}__attempt{i}.json")) for i in range(1, count + 1)]
                prompt_hash = value["selector_prompt_sha256"] if stage == "selector" else read(output / "direct_prompts" / (key + ".json"))["prompt_sha256"]
                completion.audit_original_calls(records, stage, value, query, rows, banks["gsm8k"], policy, prompt_hash)
    freeze = read(output / "prediction_freeze.json")
    assert freeze["original_reported_cost_usd"] == 7.0
    assert freeze["additional_reported_cost_usd"] == pytest.approx(0.8 * len(new_calls))
    assert freeze["reported_cost_usd"] == pytest.approx(7 + 0.8 * len(new_calls))
    completion.broad.old.previous.verify_files(output, freeze["files"])
