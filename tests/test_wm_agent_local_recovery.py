"""Synthetic format recovery and frozen/blinded workflow tests; no real calls."""

import copy
import json
from pathlib import Path

import pytest

from tools.outcome_prediction import wm_agent_local_recovery as recovery
from tools.outcome_prediction.wm_agent_local_fit import FEATURE_KEYS
from tools.outcome_prediction.wm_clean_refresh import read, sha, write


def fixture():
    pool, labels = [], {}
    for benchmark in ("gsm8k", "aime2025"):
        for i in range(8):
            values = dict.fromkeys(FEATURE_KEYS, 0.0)
            values.update({"parent.accuracy": 0.1, "parent.reference.measured": 1.0,
                           "current.config.epochs": float(i)})
            key = f"{benchmark}-train-{i // 2}/exp-{i}"
            pool.append({"example_id": key, "cell_id": f"{benchmark}-train-{i // 2}",
                         "benchmark": benchmark, "split": "train", "parent_reference": 0.1,
                         "reference_kind": "measured_parent", "views": {"history": values}})
            target = i / 20
            labels[key] = {"accuracy": target, "delta_accuracy": target - 0.1}
    query = copy.deepcopy(pool[0])
    query.update(example_id="heldout/exp-0", cell_id="heldout", split="test")
    same = [row for row in pool if row["benchmark"] == "gsm8k"]
    aliases = recovery.original.aliases_for(same)
    response = {"selected_aliases": list(aliases), "weights": [1.0] * 8,
                "model": "ridge_10", "feature_keys": ["parent.accuracy"],
                "rationale": "A frozen synthetic selection.",
                "weights_note": "This harmless extra field was rejected originally."}
    policy = {"sources": {}, "selector_system": recovery.original.SELECT_SYSTEM,
              "predictor_system": recovery.original.PREDICT_SYSTEM, "model_requested": "claude-opus-5",
              "model_menu": list(recovery.original.MODELS), "effort": "high",
              "feature_order": list(FEATURE_KEYS), "max_attempts": 2,
              "predictor_max_output_tokens": 2048, "predictor_budget_usd": 0.5,
              "max_cli_network_retries": 1, "timeout_seconds": 240}
    return pool, labels, query, aliases, response, policy


def record_for(response, *, query=None, attempt=1, stage="selector", prompt_hash="prompt-hash"):
    query = query or fixture()[2]
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-5",
         "tools": [], "mcp_servers": [], "plugins": [], "skills": []},
        {"type": "assistant", "message": {"model": "claude-opus-5",
                                             "content": [{"type": "text", "text": "synthetic"}]}},
        {"type": "result", "is_error": False, "subtype": "success",
         "result": json.dumps(response), "total_cost_usd": 0.2},
    ]
    return {"stdout": "\n".join(json.dumps(event) for event in events), "stderr": "", "returncode": 0,
            "raw_events": events, "response": response, "models_resolved": ["claude-opus-5"],
            "model_requested": "claude-opus-5", "effort": "high", "cost_usd_reported": 0.2,
            "error": "invalid_selection: extra JSON key" if stage == "selector" else None,
            "example_id": query["example_id"], "stage": stage, "attempt": attempt,
            "prompt_sha256": prompt_hash}


def recover(calls):
    pool, _, query, aliases, _, policy = fixture()
    return recovery.recover_selection(calls, aliases, pool[:8], query, policy, "prompt-hash")


def test_extra_key_projection_keeps_exact_required_values_and_ignored_provenance():
    _, _, _, _, response, _ = fixture()
    response.update(cost_usd=0.12, additional_notes={"relevant": "metadata"})
    record = record_for(response)
    before = copy.deepcopy(record)
    selected, audit = recover([record])
    assert selected["model"] == response["model"]
    assert selected["weights"] == response["weights"]
    assert selected["feature_keys"] == response["feature_keys"]
    assert audit[0]["accepted"]
    assert audit[0]["ignored_fields"] == {"weights_note": response["weights_note"],
                                          "cost_usd": 0.12, "additional_notes": {"relevant": "metadata"}}
    assert record == before


def test_first_semantically_valid_original_attempt_is_used():
    response = fixture()[4]
    first, second = copy.deepcopy(response), copy.deepcopy(response)
    first["weights"][0] = 0
    second["model"] = "ridge_100"
    selected, audit = recover([record_for(first), record_for(second, attempt=2)])
    assert selected["model"] == "ridge_100"
    assert [item["accepted"] for item in audit] == [False, True]
    first["weights"][0] = 1
    selected, audit = recover([record_for(first), record_for(second, attempt=2)])
    assert selected["model"] == "ridge_10"
    assert len(audit) == 1


@pytest.mark.parametrize("invalid", [
    "model", "tools", "tool_use", "raw_events", "response", "returncode", "effort",
    "query_identity", "stage", "attempt", "prompt", "missing_required", "invalid_weights",
    "unknown_alias", "too_few_examples",
])
def test_projection_does_not_relax_semantics_or_raw_isolation(invalid):
    response = fixture()[4]
    if invalid == "missing_required":
        del response["model"]
    elif invalid == "invalid_weights":
        response["weights"][0] = 4.1
    elif invalid == "unknown_alias":
        response["selected_aliases"][0] = "heldout"
    elif invalid == "too_few_examples":
        response["selected_aliases"] = response["selected_aliases"][:4]
        response["weights"] = response["weights"][:4]
    record = record_for(response)
    if invalid in {"model", "tools", "tool_use"}:
        if invalid == "model":
            record["raw_events"][0]["model"] = "claude-other"
        elif invalid == "tools":
            record["raw_events"][0]["tools"] = ["Read"]
        else:
            record["raw_events"][1]["message"]["content"] = [{"type": "tool_use"}]
        record["stdout"] = "\n".join(json.dumps(event) for event in record["raw_events"])
    elif invalid == "raw_events":
        record["raw_events"] = []
    elif invalid == "response":
        record["response"] = {**response, "model": "shallow_tree"}
    elif invalid == "returncode":
        record["returncode"] = 1
    elif invalid == "effort":
        record["effort"] = "low"
    elif invalid == "query_identity":
        record["example_id"] = "another-query"
    elif invalid == "stage":
        record["stage"] = "predictor"
    elif invalid == "attempt":
        record["attempt"] = 2
    elif invalid == "prompt":
        record["prompt_sha256"] = "another-prompt"
    selected, audit = recover([record])
    assert selected is None
    assert not audit[0]["accepted"]


def test_budget_reserves_inflight_limits_and_retains_reported_cost():
    budget = recovery.CallBudget(1.0, 0.5)
    assert budget.reserve() and budget.reserve()
    assert not budget.reserve()
    budget.settle({"cost_usd_reported": 0.1})
    assert not budget.reserve()  # 0.1 charged + 0.5 inflight + 0.5 next > 1
    budget.settle({"cost_usd_reported": 0.1})
    assert budget.reserve()
    budget.settle({"cost_usd_reported": 0.8})
    assert budget.reported == 1.0
    assert budget.stopped and not budget.reserve()


@pytest.mark.parametrize("cost", [None, True, "0.1", -1, float("nan"), float("inf")])
def test_unknown_or_invalid_cost_fails_closed(cost):
    budget = recovery.CallBudget(10, 0.5)
    assert budget.reserve()
    budget.settle({"cost_usd_reported": cost})
    assert budget.charged == 0.5 and budget.reported == 0
    assert budget.stopped and not budget.reserve()


def test_no_call_is_made_after_systemic_or_budget_stop(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not invoke")

    monkeypatch.setattr(recovery.io, "call_json", forbidden)
    budget = recovery.CallBudget(0.1, 0.5)
    result = recovery._invoke("prompt", {}, budget)
    assert result["attempted"] is False


def create_bundle(tmp_path):
    pool, labels, query, _, _, policy = fixture()
    successful = copy.deepcopy(query)
    successful.update(example_id="successful-heldout/exp-0", cell_id="successful-heldout")
    queries = [query, successful]
    banks = {b: recovery.original.aliases_for([row for row in pool if row["benchmark"] == b])
             for b in recovery.original.previous.TASKS}
    packets = []
    for row in queries:
        subset = [r for r in pool if r["benchmark"] == row["benchmark"]]
        subset_labels = {r["example_id"]: labels[r["example_id"]] for r in subset}
        prompt = recovery.original.selector_prompt(row, subset, subset_labels, banks[row["benchmark"]], policy["feature_order"])
        packets.append({"example_id": row["example_id"], "benchmark": row["benchmark"],
                        "prompt": prompt, "prompt_sha256": recovery.original.digest(prompt)})
    bundle, source, output = (tmp_path / name for name in ("bundle", "original", "recovery"))
    bundle.mkdir()
    source.mkdir()
    for name, value in (("train_inputs.json", pool), ("train_labels.json", labels), ("queries.json", queries),
                        ("aliases.json", banks), ("selector_prompts.json", packets), ("policy.json", policy)):
        write(bundle / name, value)
    write(bundle / "manifest.json", {path.name: sha(path) for path in bundle.iterdir()})
    return bundle, source, output, pool, labels, queries, packets


def create_original_results(source, queries, packets, *, invalid=False):
    for name in ("calls", "query_results"):
        (source / name).mkdir()
    results = []
    response = fixture()[4]
    if invalid:
        response["weights"][0] = 0
    for i, query in enumerate(queries):
        result = {"example_id": query["example_id"], "benchmark": query["benchmark"],
                  "selector_prompt_sha256": packets[i]["prompt_sha256"],
                  "selection": None, "local_fit": None, "median_fit": None,
                  "llm_prediction": None, "error": "invalid_selection: exact schema", "selector_attempts": 1}
        key = recovery.original.key_for(query)
        if i:
            result.update(selection={"original": "unchanged successful selection"},
                          local_fit={"predicted_accuracy": 0.5}, median_fit={"predicted_accuracy": 0.4},
                          llm_prediction={"predicted_accuracy": 0.3, "rationale": "Original forecast"},
                          error=None, predictor_attempts=1)
        else:
            call = record_for(response, query=query, prompt_hash=packets[i]["prompt_sha256"])
            write(source / "calls" / (key + "__selector__attempt1.json"), call)
        results.append(result)
        write(source / "query_results" / (key + ".json"), result)
    write(source / "predictions.json", results)
    write(source / "prediction_freeze.json", {"created_at": recovery.original.now(),
                                              "files": {str(path.relative_to(source)): sha(path)
                                                        for path in source.rglob("*") if path.is_file()}})
    return results


def test_prepare_does_not_open_predictions_or_any_labels(monkeypatch, tmp_path):
    bundle, source, output, *_ = create_bundle(tmp_path)
    reads = []

    def tracked(path):
        reads.append(Path(path).name)
        assert "label" not in Path(path).name and "prediction" not in Path(path).name
        return read(path)

    monkeypatch.setattr(recovery, "read", tracked)
    recovery.prepare(bundle, source, output)
    assert reads == ["manifest.json"]
    policy = read(output / "recovery_policy.json")
    assert policy["max_predictor_attempts"] == 2
    assert policy["reported_cost_stop_usd"] == 10
    assert any(path.endswith("wm_agent_local_recovery_score.py") for path in policy["sources"])
    with pytest.raises(FileExistsError):
        recovery.prepare(bundle, source, output)


def test_run_requires_original_freeze_and_unchanged_sources(tmp_path):
    bundle, source, output, *_ = create_bundle(tmp_path)
    recovery.prepare(bundle, source, output)
    with pytest.raises(FileNotFoundError):
        recovery.run(output)


@pytest.mark.parametrize("invalid", [False, True])
def test_blinded_recovery_preserves_original_success_and_only_calls_predictor(monkeypatch, tmp_path, invalid):
    bundle, source, output, _, _, queries, packets = create_bundle(tmp_path)
    recovery.prepare(bundle, source, output)
    originals = create_original_results(source, queries, packets, invalid=invalid)
    calls, fits = [], []

    def guarded_read(path):
        assert Path(path).name != "labels.json", "Global target labels must never be opened"
        return read(path)

    def fake_fit(rows, labels, query, selection):
        assert set(labels) == {row["example_id"] for row in rows}
        assert query["example_id"] not in labels
        assert all(row["split"] == "train" and row["cell_id"] != query["cell_id"] for row in rows)
        fits.append(selection["model"])
        return {"predicted_accuracy": 0.2, "fit_metadata": {"normalized_sample_weights": [1.0] * len(rows)}}

    def fake_call(prompt, policy):
        assert policy["system"] == recovery.original.PREDICT_SYSTEM
        assert policy["max_output_tokens"] == 2048
        payload = json.loads(prompt)
        assert "labeled_training_demonstrations" in payload and "training_pool" not in payload
        calls.append(payload)
        return record_for({"predicted_accuracy": 0.25, "rationale": "Synthetic forecast"}, stage="predictor")

    monkeypatch.setattr(recovery, "read", guarded_read)
    monkeypatch.setattr(recovery, "fit_predict", fake_fit)
    monkeypatch.setattr(recovery.io, "call_json", fake_call)
    result = recovery.run(output)
    merged = read(output / "predictions.json")
    by_id = {row["example_id"]: row for row in merged}
    assert by_id[queries[1]["example_id"]] == originals[1]
    kept_name = recovery.original.key_for(queries[1]) + ".json"
    assert sha(output / "query_results" / kept_name) == sha(source / "query_results" / kept_name)
    provenance = read(output / "recovery_provenance.json")
    failed = provenance[queries[0]["example_id"]]
    assert failed["attempts"][0]["original_call_sha256"]
    if invalid:
        assert not calls and not fits
        assert result["recovered"] == 0
        assert by_id[queries[0]["example_id"]] == originals[0]
        assert failed["mode"] == "unrecoverable_original_failure"
    else:
        assert len(calls) == 1 and fits == ["ridge_10", "weighted_median"]
        assert result["recovered"] == 1 and result["reported_cost_usd"] == 0.2
        assert failed["mode"] == "recovered_extra_key_projection"
        assert failed["attempts"][0]["ignored_fields"] == {"weights_note": fixture()[4]["weights_note"]}
    freeze = read(output / "prediction_freeze.json")
    recovery.original.previous.verify_files(output, freeze["files"])
    assert "recovery_provenance.json" in freeze["files"]
    assert "predictions.json" in freeze["files"]


def test_prepare_rejects_already_scored_original(tmp_path):
    bundle, source, output, *_ = create_bundle(tmp_path)
    write(source / "report.json", {"already": "scored"})
    with pytest.raises(ValueError, match="before original target scoring"):
        recovery.prepare(bundle, source, output)
