"""Blinded additive recovery of otherwise valid selections with extra JSON keys.

``prepare`` freezes this recovery policy and sources before any original forecasts
are opened. ``run`` requires the original prediction freeze, never reads a target
label file, and only recovers failed selections from already generated tool-free
responses. Original non-null selections/results are preserved without reconsidering
earlier attempts. No scoring or new selector calls are provided by this module.
"""

from __future__ import annotations

import argparse
import copy
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.outcome_prediction import wm_agent_local_benchmark as original
from tools.outcome_prediction import wm_local_agent_io as io
from tools.outcome_prediction.wm_agent_local_fit import _validate_row, fit_predict
from tools.outcome_prediction.wm_clean_refresh import read, sha, write

OUTPUT = Path("data/analysis/wm_agent_local/v1_recovery")
REQUIRED = frozenset({"selected_aliases", "weights", "model", "feature_keys", "rationale"})
RULES = {
    "schema": "blinded-extra-key-selection-recovery-v1", "workers": 4,
    "max_predictor_attempts": 2, "reported_cost_stop_usd": 10.0,
    "selection": "Preserve every original non-null selection/result. For failed selections only, inspect original attempts in order; verify raw exact-model tool-free call, project exactly the five required fields, validate unchanged original constraints, accept first semantically valid attempt. No new selector calls or fit/test feedback.",
    "ignored_fields": "Record every ignored key and value plus original call path/hash; never reinterpret or use ignored metadata.",
    "forecast": "Fit original model and median on selected TRAIN rows only; original direct-LLM system, selected-example prompt and invocation settings. At most two fresh predictor calls. No forecast overrides.",
    "budget": "Reserve original per-call budget before each invocation; stop scheduling when charged cost plus concurrent reservations reaches $10. Missing/invalid cost charges full reservation and stops calls. CLI-reported cost is not subscription billing and may exceed its configured limit slightly.",
    "blindness": "No test target labels opened. Recovery policy/sources frozen before original predictions are loaded. Original prediction_freeze required before recovery. No scoring in this module.",
}
STOP_TERMS = (
    "rate limit", "rate_limit", "usage limit", "session limit", "hit your limit",
    "authentication", "model identity", "unexpected tools", "billing", "credit balance",
)


def _source_hashes():
    names = (
        "wm_agent_local_recovery.py", "wm_agent_local_benchmark.py", "wm_agent_local_fit.py",
        "wm_local_agent_io.py", "wm_fresh_llm_baseline.py", "wm_clean_refresh.py",
        "wm_one_step_train.py", "wm_one_step_features.py",
        "wm_agent_local_recovery_score.py",
    )
    return {str((Path(__file__).parent / name).resolve()): sha(Path(__file__).parent / name)
            for name in names}


def prepare(bundle=original.BUNDLE, original_results=original.OUTPUT, output=OUTPUT):
    """Freeze rules first; no prediction, query outcome, or source-label reads."""
    bundle, original_results, output = map(Path, (bundle, original_results, output))
    if output.exists():
        raise FileExistsError("Recovery artifacts must use a new immutable directory")
    if (original_results / "report.json").exists():
        raise ValueError("Recovery must be designed before original target scoring")
    manifest = read(bundle / "manifest.json")
    original.previous.verify_files(bundle, manifest)
    policy = {
        **RULES, "created_at": original.now(), "bundle": str(bundle.resolve()),
        "bundle_manifest_sha256": sha(bundle / "manifest.json"),
        "original_results": str(original_results.resolve()), "sources": _source_hashes(),
    }
    output.mkdir(parents=True, mode=0o700)
    write(output / "recovery_policy.json", policy)
    write(output / "recovery_policy_freeze.json", {
        "created_at": original.now(), "files": {"recovery_policy.json": sha(output / "recovery_policy.json")},
        "note": "Frozen without opening original predictions or any query target labels.",
    })
    return {"output": str(output), "policy_sha256": sha(output / "recovery_policy.json")}


def load_blinded_bundle(bundle):
    """Verify the already frozen input-only query and TRAIN-only label bundle."""
    bundle = Path(bundle)
    manifest = read(bundle / "manifest.json")
    required = {"train_inputs.json", "train_labels.json", "queries.json", "aliases.json",
                "selector_prompts.json", "policy.json"}
    if set(manifest) != required:
        raise ValueError("Unexpected blinded input bundle files")
    original.previous.verify_files(bundle, manifest)
    policy = read(bundle / "policy.json")
    for path, expected in policy["sources"].items():
        if sha(path) != expected:
            raise ValueError("Changed original frozen source")
    if (policy["selector_system"] != original.SELECT_SYSTEM
            or policy["predictor_system"] != original.PREDICT_SYSTEM
            or policy["model_requested"] != "claude-opus-5"
            or policy["model_menu"] != list(original.MODELS)):
        raise ValueError("Original selection/inference policy changed")
    train, labels, queries, banks, packets = (
        read(bundle / name) for name in (
            "train_inputs.json", "train_labels.json", "queries.json", "aliases.json", "selector_prompts.json",
        )
    )
    original.validate_rows(train, labels)
    original.validate_rows(queries)
    for row in train:
        _validate_row(row, training=True)
    for query in queries:
        _validate_row(query, training=False)
        if query["split"] != "test":
            raise ValueError("Recovery query must be an original test input")
    if {r["cell_id"] for r in train} & {q["cell_id"] for q in queries}:
        raise ValueError("Training/query session overlap")
    if len(packets) != len(queries):
        raise ValueError("Query packet count mismatch")
    expected_banks = {b: original.aliases_for([r for r in train if r["benchmark"] == b])
                      for b in original.previous.TASKS}
    if banks != expected_banks:
        raise ValueError("Changed original training aliases")
    for query, packet in zip(queries, packets):
        pool = [r for r in train if r["benchmark"] == query["benchmark"]]
        prompt = original.selector_prompt(query, pool, {r["example_id"]: labels[r["example_id"]] for r in pool},
                                          banks[query["benchmark"]], policy["feature_order"])
        expected = {"example_id": query["example_id"], "benchmark": query["benchmark"],
                    "prompt": prompt, "prompt_sha256": original.digest(prompt)}
        if packet != expected:
            raise ValueError("Changed train-only selector packet")
    return policy, train, labels, queries, banks, packets


def recover_selection(calls, aliases, pool, query, policy, prompt_sha256):
    """Pure ordered raw-call validation and extra-key projection; no fitting."""
    audit = []
    for attempt, record in enumerate(calls, 1):
        item = {"attempt": attempt, "accepted": False, "ignored_fields": {}}
        try:
            if (record.get("stage") != "selector" or record.get("attempt") != attempt
                    or record.get("example_id") != query["example_id"]
                    or record.get("prompt_sha256") != prompt_sha256):
                raise ValueError("Original call identity/attempt/prompt mismatch")
            # This re-decodes stdout and ignores only the later exact-schema
            # rejection; model identity, tool isolation and raw JSON stay strict.
            original.validate_success_call(record, policy)
            response = record["response"]
            if not isinstance(response, dict) or not REQUIRED.issubset(response):
                raise ValueError("Original response lacks required selection fields")
            item["ignored_fields"] = {key: value for key, value in response.items() if key not in REQUIRED}
            projected = {key: response[key] for key in REQUIRED}
            selection = original.resolve_selection(projected, aliases, pool, query)
            item.update(accepted=True, reason="First semantically valid original response after five-key projection")
            audit.append(item)
            return selection, audit
        except (ValueError, TypeError, KeyError) as exc:
            item["reason"] = str(exc)
            audit.append(item)
    return None, audit


class CallBudget:
    """Concurrent budget reservations; missing accounting closes further calls."""

    def __init__(self, limit, reservation):
        self.limit, self.reservation = float(limit), float(reservation)
        self.reported, self.charged, self.reserved = 0.0, 0.0, 0.0
        self.stopped = False
        self.lock = threading.Lock()

    def reserve(self):
        with self.lock:
            if self.stopped or self.charged + self.reserved + self.reservation > self.limit + 1e-12:
                return False
            self.reserved += self.reservation
            return True

    def settle(self, record):
        with self.lock:
            self.reserved -= self.reservation
            cost = record.get("cost_usd_reported")
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                self.charged += self.reservation
                self.stopped = True
            else:
                self.reported += cost
                self.charged += cost
            if self.charged >= self.limit or any(term in str(record.get("error")).lower() for term in STOP_TERMS):
                self.stopped = True


def _invoke(prompt, policy, budget):
    if not budget.reserve():
        return {"response": None, "error": "unattempted_after_recovery_budget_or_systemic_stop",
                "attempted": False, "created_at": original.now()}
    try:
        record = io.call_json(prompt, policy)
        record["attempted"] = True
    except Exception:  # Release reservation and fail closed before surfacing a programming failure.
        budget.settle({"cost_usd_reported": None})
        raise
    budget.settle(record)
    return record


def run(output=OUTPUT):
    output = Path(output)
    recovery_freeze = read(output / "recovery_policy_freeze.json")
    original.previous.verify_files(output, recovery_freeze["files"])
    recovery = read(output / "recovery_policy.json")
    if any(recovery.get(key) != value for key, value in RULES.items()) or recovery["sources"] != _source_hashes():
        raise ValueError("Changed frozen recovery rules/source")
    bundle, source = Path(recovery["bundle"]), Path(recovery["original_results"])
    if sha(bundle / "manifest.json") != recovery["bundle_manifest_sha256"]:
        raise ValueError("Changed original input bundle")
    if (source / "report.json").exists():
        raise ValueError("Recovery cannot begin after original target scoring")
    freeze = read(source / "prediction_freeze.json")
    original.previous.verify_files(source, freeze["files"])
    policy, train, labels, queries, banks, packets = load_blinded_bundle(bundle)
    # All verification above uses hashes, input-only queries and TRAIN labels.
    predictions = read(source / "predictions.json")
    by_id = {p["example_id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {q["example_id"] for q in queries}:
        raise ValueError("Original prediction identities mismatch")
    for name in ("calls", "selected_prompts", "query_results"):
        (output / name).mkdir(mode=0o700, exist_ok=False)
    write(output / "run_policy.json", {
        "started_at": original.now(), "original_prediction_freeze_sha256": sha(source / "prediction_freeze.json"),
        "original_prediction_frozen_at": freeze["created_at"],
        "recovery_policy_freeze_sha256": sha(output / "recovery_policy_freeze.json"),
    })
    predictor_policy = {
        **policy, "system": policy["predictor_system"],
        "max_output_tokens": policy["predictor_max_output_tokens"],
        "per_call_budget_usd": policy["predictor_budget_usd"],
    }
    budget = CallBudget(recovery["reported_cost_stop_usd"], policy["predictor_budget_usd"])
    train_by_id = {row["example_id"]: row for row in train}
    packet_of = {packet["example_id"]: packet for packet in packets}
    started = time.perf_counter()

    def work(query):
        key, example_id = original.key_for(query), query["example_id"]
        original_path = source / "query_results" / (key + ".json")
        if str(original_path.relative_to(source)) not in freeze["files"]:
            raise ValueError("Original query result was not frozen")
        result = copy.deepcopy(by_id[example_id])
        if result != read(original_path):
            raise ValueError("Original query result differs from frozen predictions")
        provenance = {"mode": "preserved_original", "original_query_result_path": str(original_path.resolve()),
                      "original_query_result_sha256": sha(original_path), "attempts": []}
        if result["selection"] is not None:
            write(output / "query_results" / (key + ".json"), result)
            return result, provenance
        calls, call_paths = [], []
        count = result.get("selector_attempts")
        if type(count) is not int or not 1 <= count <= policy["max_attempts"]:
            raise ValueError("Invalid original selector attempt count")
        for attempt in range(1, count + 1):
            path = source / "calls" / (key + f"__selector__attempt{attempt}.json")
            if str(path.relative_to(source)) not in freeze["files"]:
                raise ValueError("Original selector call was not frozen")
            calls.append(read(path))
            call_paths.append(path)
        pool = [row for row in train if row["benchmark"] == query["benchmark"]]
        selection, audit = recover_selection(calls, banks[query["benchmark"]], pool, query, policy,
                                             packet_of[example_id]["prompt_sha256"])
        for item, path in zip(audit, call_paths):
            item.update(original_call_path=str(path.resolve()), original_call_sha256=sha(path))
        provenance["attempts"] = audit
        if selection is None:
            provenance["mode"] = "unrecoverable_original_failure"
            write(output / "query_results" / (key + ".json"), result)
            return result, provenance
        provenance["mode"] = "recovered_extra_key_projection"
        chosen = [train_by_id[example] for example in selection["selected_ids"]]
        chosen_labels = {row["example_id"]: labels[row["example_id"]] for row in chosen}
        result.update(selection=selection, selector_attempts=audit[-1]["attempt"],
                      local_fit=None, median_fit=None, llm_prediction=None, error=None)
        try:
            local = fit_predict(chosen, chosen_labels, query, selection)
            median = fit_predict(chosen, chosen_labels, query, {**selection, "model": "weighted_median", "feature_keys": []})
            result.update(local_fit=local, median_fit=median)
            weights = local["fit_metadata"]["normalized_sample_weights"]
            prompt = original.selected_prompt(query, chosen, chosen_labels, weights, policy["feature_order"])
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            result["error"] = "recovery_fit_error: " + str(exc)
            write(output / "query_results" / (key + ".json"), result)
            return result, provenance
        prompt_hash = original.digest(prompt)
        write(output / "selected_prompts" / (key + ".json"), {
            "example_id": example_id, "prompt": prompt, "prompt_sha256": prompt_hash,
            "selected_ids": selection["selected_ids"], "effective_weights": weights,
        })
        for attempt in range(1, recovery["max_predictor_attempts"] + 1):
            record = _invoke(prompt, predictor_policy, budget)
            if record.get("response") is not None and not record.get("error"):
                try:
                    original.validate_success_call(record, policy)
                    result["llm_prediction"] = original.previous.parse_prediction(original.canonical(record["response"]))
                except (ValueError, TypeError, KeyError) as exc:
                    record["error"] = "invalid_forecast: " + str(exc)
            record.update(example_id=example_id, stage="predictor", attempt=attempt, prompt_sha256=prompt_hash)
            write(output / "calls" / (key + f"__predictor__attempt{attempt}.json"), record)
            result["predictor_attempts"] = attempt
            if result["llm_prediction"] is not None or not record.get("attempted") or budget.stopped:
                break
        result["error"] = None if result["llm_prediction"] is not None else record["error"]
        write(output / "query_results" / (key + ".json"), result)
        return result, provenance

    merged, provenance = [], {}
    with ThreadPoolExecutor(max_workers=recovery["workers"]) as executor:
        pending = [executor.submit(work, query) for query in queries]
        for future in as_completed(pending):
            result, audit = future.result()
            merged.append(result)
            provenance[result["example_id"]] = audit
            print(f"recovery {len(merged)}/{len(queries)} {result['example_id']} {audit['mode']} status={result['error'] or 'ok'}", flush=True)
    merged.sort(key=lambda row: row["example_id"])
    write(output / "predictions.json", merged)
    write(output / "recovery_provenance.json", provenance)
    write(output / "prediction_freeze.json", {
        "created_at": original.now(), "elapsed_seconds": time.perf_counter() - started,
        "reported_cost_usd": budget.reported, "charged_cost_usd": budget.charged,
        "systemic_stop": budget.stopped,
        "files": {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
        "note": "Merged original results and format-recovered forecasts frozen without opening held-out target labels. Original successes unchanged; no new selector calls.",
    })
    return {"queries": len(merged), "recovered": sum(p["mode"] == "recovered_extra_key_projection" for p in provenance.values()),
            "local_success": sum(row["local_fit"] is not None for row in merged),
            "llm_success": sum(row["llm_prediction"] is not None for row in merged), "reported_cost_usd": budget.reported}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--bundle", type=Path, default=original.BUNDLE)
    parser.add_argument("--original-results", type=Path, default=original.OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = prepare(args.bundle, args.original_results, args.output) if args.action == "prepare" else run(args.output)
    print(original.canonical(value))


if __name__ == "__main__":
    main()
