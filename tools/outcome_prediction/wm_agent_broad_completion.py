"""Blinded higher-token/budget completion sensitivity for frozen V2 forecasts.

Prepare the additive policy before opening original predictions. After the strict
prediction freeze, preserve every successful original stage and only append up
to two calls for missing stages. Systems, inputs, model identity, parser, numeric
fits and controls remain fixed; higher token/budget limits can affect reasoning,
so this is an execution-allowance sensitivity, not identical inference sampling.
No target scoring or global held-out label reads are implemented here.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.outcome_prediction import wm_agent_broad_benchmark as broad
from tools.outcome_prediction import wm_agent_local_recovery as recovery
from tools.outcome_prediction.wm_agent_broad_fit import fit_predict
from tools.outcome_prediction.wm_clean_refresh import read, sha, write

OUTPUT = Path("data/analysis/wm_agent_local/v2_completion")
RULES = {
    "schema": "blinded-broad-v2-higher-allowance-completion-v1",
    "workers": 4, "extra_attempts_per_stage": 2, "max_output_tokens": 4096,
    "per_call_budget_usd": 2.0, "additional_reported_cost_stop_usd": 40.0,
    "preservation": "All original valid specs, numeric fits and successful forecasts unchanged. Original calls retain attempt numbers and bytes; additional calls append after original counters. Complete original frozen snapshots retained.",
    "selection": "Only missing specs may receive new selector calls. Same exact original system, frozen packet, aliases, feature schema, parser and selection constraints. First valid chronological call, never best-of or model-quality feedback.",
    "prediction": "Only missing forecasts receive new predictor calls, using exact original full-pool factual prompt and fitted weights. Preserve existing fits; generate missing fits/control arms with unchanged backend/spec only. No query labels or validation feedback.",
    "allowance_sensitivity": "Only output allowance and per-call/additional execution budgets change. 4096 tokens versus the original predictor2048 can change reasoning; this is higher-allowance sensitivity, not merely format recovery or identical inference.",
    "budget": "Additional-only $40 reservation cap, $2 reserved per call, four workers; actual CLI-reported costs can slightly exceed configured limits. Missing cost accounting stops further scheduling. Full reported cost includes original plus additional phases.",
}


def source_hashes():
    return {**broad.source_hashes(), str(Path(__file__).resolve()): sha(Path(__file__))}


def prepare(bundle=broad.BUNDLE, original_results=broad.OUTPUT, output=OUTPUT):
    """Freeze operational sensitivity before any original predictions are opened."""
    bundle, original_results, output = map(Path, (bundle, original_results, output))
    if output.exists():
        raise FileExistsError("Completion needs a new immutable directory")
    if (original_results / "report.json").exists():
        raise ValueError("Prepare completion before original target scoring")
    broad.old.previous.verify_files(bundle, read(bundle / "manifest.json"))
    policy = {**RULES, "created_at": broad.old.now(), "bundle": str(bundle.resolve()),
              "bundle_manifest_sha256": sha(bundle / "manifest.json"),
              "original_results": str(original_results.resolve()), "sources": source_hashes()}
    output.mkdir(parents=True, mode=0o700)
    write(output / "completion_policy.json", policy)
    write(output / "completion_policy_freeze.json", {
        "created_at": broad.old.now(), "files": {"completion_policy.json": sha(output / "completion_policy.json")},
        "note": "Policy/source frozen before opening any original V2 predictions or query outcome labels.",
    })
    return {"output": str(output), "policy_sha256": sha(output / "completion_policy.json")}


def copy_frozen_file(source, destination, expected):
    """Explicit new-file copy with before/after hashes; never overwrite originals."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not source.is_file() or sha(source) != expected:
        raise ValueError("Original frozen file changed before copy")
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)
    if sha(destination) != expected:
        raise ValueError("Frozen snapshot copy mismatch")


def audit_original_calls(records, stage, result, query, pool, aliases, policy, prompt_hash):
    """Verify original chronological validity without using an errored partial result."""
    accepted = []
    for index, record in enumerate(records, 1):
        if (record.get("example_id") != query["example_id"] or record.get("stage") != stage
                or record.get("attempt") != index or record.get("prompt_sha256") != prompt_hash):
            raise ValueError("Original stage call identity/attempt/prompt mismatch")
        value = None
        try:
            broad.old.validate_success_call(record, policy)
            if stage == "selector":
                value, ignored = broad.resolve_spec(record["response"], aliases, pool, query)
                if ignored != record.get("ignored_fields"):
                    raise ValueError("Original ignored metadata mismatch")
            else:
                value = broad.old.previous.parse_prediction(broad.old.canonical(record["response"]))
        except (ValueError, TypeError, KeyError):
            if not record.get("error"):
                raise ValueError("Original success flag fails raw/model/schema audit") from None
            continue
        if record.get("error"):
            raise ValueError("Original error flag hides a semantically valid response; not execution-only completion")
        accepted.append((index, value))
    expected = result["spec"] if stage == "selector" else result["llm_prediction"]
    if expected is None:
        if accepted:
            raise ValueError("Original missing result had a valid earlier response")
    elif len(accepted) != 1 or accepted[0][0] != len(records) or accepted[0][1] != expected:
        raise ValueError("Original saved stage is not its first valid chronological response")


def run(output=OUTPUT):
    output = Path(output)
    policy_freeze = read(output / "completion_policy_freeze.json")
    broad.old.previous.verify_files(output, policy_freeze["files"])
    completion = read(output / "completion_policy.json")
    if any(completion.get(key) != value for key, value in RULES.items()) or completion["sources"] != source_hashes():
        raise ValueError("Changed frozen completion policy/source")
    bundle, source = Path(completion["bundle"]), Path(completion["original_results"])
    if sha(bundle / "manifest.json") != completion["bundle_manifest_sha256"]:
        raise ValueError("Changed original V2 bundle")
    if (source / "report.json").exists():
        raise ValueError("Completion cannot begin after original target scoring")
    strict_freeze = read(source / "prediction_freeze.json")
    frozen = strict_freeze["files"]
    broad.old.previous.verify_files(source, frozen)
    for required in ("predictions.json", "run_policy.json"):
        if required not in frozen:
            raise ValueError("Incomplete original forecast freeze")
    # Frozen V2 validation reads input-only queries, training labels and hashes;
    # it does not open the held-out accuracy label table or any metric report.
    policy, train, labels, queries, banks, evidence, packets = broad.validate_bundle(bundle)
    original_run = read(source / "run_policy.json")
    if original_run["bundle_manifest_sha256"] != completion["bundle_manifest_sha256"]:
        raise ValueError("Original run used a different bundle")
    originals = read(source / "predictions.json")
    by_id = {row["example_id"]: row for row in originals}
    if len(by_id) != len(originals) or set(by_id) != {query["example_id"] for query in queries}:
        raise ValueError("Original query membership mismatch")
    snapshot = output / "original_artifacts"
    snapshot.mkdir(mode=0o700, exist_ok=False)
    for name, expected in frozen.items():
        if name in {"report.json", "manifest.json", "prediction_freeze.json"}:
            raise ValueError("Original freeze unexpectedly includes scored/self artifacts")
        copy_frozen_file(source / name, snapshot / name, expected)
        if name not in {"predictions.json"} and not name.startswith("query_results/"):
            copy_frozen_file(source / name, output / name, expected)
    copy_frozen_file(source / "prediction_freeze.json", snapshot / "prediction_freeze.json", sha(source / "prediction_freeze.json"))
    for name in ("calls", "direct_prompts", "query_results"):
        (output / name).mkdir(mode=0o700, exist_ok=True)
    write(output / "completion_run_policy.json", {
        "started_at": broad.old.now(), "original_prediction_freeze_path": str((source / "prediction_freeze.json").resolve()),
        "original_prediction_freeze_sha256": sha(source / "prediction_freeze.json"),
        "original_prediction_frozen_at": strict_freeze["created_at"],
        "completion_policy_freeze_sha256": sha(output / "completion_policy_freeze.json"),
        "original_reported_cost_usd": strict_freeze["reported_cost_usd"],
    })
    packet_of = {packet["example_id"]: packet for packet in packets}
    budget = recovery.CallBudget(completion["additional_reported_cost_stop_usd"], completion["per_call_budget_usd"])
    started = time.perf_counter()

    def invoke(prompt, stage):
        call_policy = {**policy, "system": policy[stage + "_system"],
                       "max_output_tokens": completion["max_output_tokens"],
                       "per_call_budget_usd": completion["per_call_budget_usd"]}
        return recovery._invoke(prompt, call_policy, budget)

    def work(query):
        identity, key = query["example_id"], broad.old.key_for(query)
        result = copy.deepcopy(by_id[identity])
        original_name = "query_results/" + key + ".json"
        if original_name not in frozen or result != read(source / original_name):
            raise ValueError("Original query result is not frozen/aligned")
        packet = packet_of[identity]
        if result["selector_prompt_sha256"] != packet["prompt_sha256"]:
            raise ValueError("Original selector packet mismatch")
        pool = [row for row in train if row["benchmark"] == query["benchmark"]]
        pool_labels = {row["example_id"]: labels[row["example_id"]] for row in pool}
        provenance = {"original_query_result_path": str((source / original_name).resolve()),
                      "original_query_result_sha256": frozen[original_name],
                      "original_selector_attempts": result.get("selector_attempts", 0),
                      "original_predictor_attempts": result.get("predictor_attempts", 0),
                      "preserved_spec": result["spec"] is not None,
                      "preserved_fits": result["fits"] is not None,
                      "preserved_forecast": result["llm_prediction"] is not None,
                      "new_calls": []}

        def existing(stage, prompt_hash):
            count = result.get(stage + "_attempts", 0)
            if type(count) is not int or not 0 <= count <= policy["max_attempts"]:
                raise ValueError("Invalid original attempt count")
            records = []
            for attempt in range(1, count + 1):
                name = "calls/" + key + f"__{stage}__attempt{attempt}.json"
                if name not in frozen:
                    raise ValueError("Original stage call missing from freeze")
                records.append(read(source / name))
            audit_original_calls(records, stage, result, query, pool, banks[query["benchmark"]], policy, prompt_hash)
            return count

        def append_stage(stage, prompt, prompt_hash, old_count):
            for attempt in range(old_count + 1, old_count + completion["extra_attempts_per_stage"] + 1):
                record = invoke(prompt, stage)
                if record.get("response") is not None and not record.get("error"):
                    try:
                        broad.old.validate_success_call(record, policy)
                        if stage == "selector":
                            result["spec"], record["ignored_fields"] = broad.resolve_spec(record["response"], banks[query["benchmark"]], pool, query)
                        else:
                            result["llm_prediction"] = broad.old.previous.parse_prediction(broad.old.canonical(record["response"]))
                    except (ValueError, TypeError, KeyError) as exc:
                        record["error"] = "invalid_" + stage + ": " + str(exc)
                record.update(example_id=identity, stage=stage, attempt=attempt, prompt_sha256=prompt_hash)
                name = "calls/" + key + f"__{stage}__attempt{attempt}.json"
                write(output / name, record)
                provenance["new_calls"].append({"stage": stage, "attempt": attempt, "path": name,
                                                "sha256": sha(output / name), "attempted": record.get("attempted", False)})
                result[stage + "_attempts"] = attempt
                success = result["spec"] is not None if stage == "selector" else result["llm_prediction"] is not None
                if success or not record.get("attempted") or budget.stopped:
                    break
            return record

        def finish():
            if result == by_id[identity]:
                copy_frozen_file(source / original_name, output / original_name, frozen[original_name])
                provenance["mode"] = "preserved_original"
            else:
                write(output / original_name, result)
                provenance["mode"] = "higher_allowance_completion"
            return result, provenance

        count = existing("selector", packet["prompt_sha256"])
        if result["spec"] is None:
            record = append_stage("selector", packet["prompt"], packet["prompt_sha256"], count)
            if result["spec"] is None:
                result["error"] = record.get("error") or "no_valid_spec_after_completion"
                return finish()
        spec = result["spec"]
        if result["fits"] is None:
            try:
                local = fit_predict(pool, pool_labels, query, spec)
                fits = {"agent_target": local}
                for target in ("accuracy", "delta"):
                    fits["fixed_" + target] = local if spec["target"] == target else fit_predict(pool, pool_labels, query, {**spec, "target": target})
                fits["uniform_weights"] = fit_predict(pool, pool_labels, query, {**spec, "weights": [1.0] * len(pool)})
                result["fits"] = fits
            except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
                result["error"] = "fit_error: " + str(exc)
                return finish()
        payload = broad.prompt_payload(query, pool, pool_labels, banks[query["benchmark"]], evidence, policy["feature_order"])
        weights = result["fits"]["agent_target"]["fit_metadata"]["normalized_sample_weights"]
        prompt = broad.direct_prompt(payload, pool, banks[query["benchmark"]], weights)
        prompt_hash = broad.old.digest(prompt)
        prompt_value = {"example_id": identity, "prompt": prompt, "prompt_sha256": prompt_hash}
        prompt_path = output / "direct_prompts" / (key + ".json")
        if prompt_path.exists():
            if read(prompt_path) != prompt_value:
                raise ValueError("Original direct factual prompt/weights changed")
        else:
            write(prompt_path, prompt_value)
        count = existing("predictor", prompt_hash)
        if result["llm_prediction"] is None:
            record = append_stage("predictor", prompt, prompt_hash, count)
            result["error"] = None if result["llm_prediction"] is not None else record.get("error")
        return finish()

    results, provenance = [], {}
    with ThreadPoolExecutor(max_workers=completion["workers"]) as executor:
        for future in as_completed([executor.submit(work, query) for query in queries]):
            result, entry = future.result()
            results.append(result)
            provenance[result["example_id"]] = entry
            print(f"completion {len(results)}/{len(queries)} {result['example_id']} {entry['mode']} status={result['error'] or 'ok'} additional={budget.reported:.2f}", flush=True)
    results.sort(key=lambda row: row["example_id"])
    write(output / "predictions.json", results)
    write(output / "completion_provenance.json", provenance)
    write(output / "prediction_freeze.json", {
        "created_at": broad.old.now(), "elapsed_seconds": time.perf_counter() - started,
        "reported_cost_usd": strict_freeze["reported_cost_usd"] + budget.reported,
        "original_reported_cost_usd": strict_freeze["reported_cost_usd"],
        "additional_reported_cost_usd": budget.reported,
        "additional_charged_cost_usd": budget.charged, "systemic_stop": budget.stopped,
        "files": {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
        "note": "Higher-token/budget completion sensitivity, with unchanged original successful stages and complete strict snapshot. Frozen before target scoring; no test labels read by completion.",
    })
    return {"queries": len(results), "local_success": sum(row["fits"] is not None for row in results),
            "llm_success": sum(row["llm_prediction"] is not None for row in results),
            "additional_reported_cost_usd": budget.reported,
            "total_reported_cost_usd": strict_freeze["reported_cost_usd"] + budget.reported}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--bundle", type=Path, default=broad.BUNDLE)
    parser.add_argument("--original-results", type=Path, default=broad.OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.bundle, args.original_results, args.output) if args.action == "prepare" else run(args.output)
    print(broad.old.canonical(result))


if __name__ == "__main__":
    main()
