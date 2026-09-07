"""Agent-selected data/model versus direct inference on the same clean test queries.

New artifacts only. Selection sees benchmark-matched TRAIN labels; per-query
model fitting and inference never receive target outcomes. No test-score tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tools.outcome_prediction import wm_fresh_llm_baseline as previous
from tools.outcome_prediction.wm_clean_refresh import read, sha, write
from tools.outcome_prediction.wm_code_benchmark import paired_comparison
from tools.outcome_prediction.wm_one_step_train import validate_bundle, validate_rows
from tools.outcome_prediction.wm_small_benchmark import evaluate

TRAINING = Path("data/analysis/wm_one_step_predictors/v1_bundle")
PREVIOUS = Path("data/analysis/wm_fresh_llm/v1_results")
REGRESSORS = Path("data/analysis/wm_one_step_predictors/v1_results")
BUNDLE = Path("data/analysis/wm_agent_local/v1_bundle")
OUTPUT = Path("data/analysis/wm_agent_local/v1_results")
SEED = "agent-local-v1-20260906"
MODELS = ("weighted_median", "ridge_10", "ridge_100", "shallow_tree")
NEW_ARMS = ("agent_local_model", "selected_few_shot", "selected_weighted_median")
SELECT_SYSTEM = """You select evidence and a small statistical model for forecasting
the outcome of a proposed post-training experiment. You have no tools and must
not output an accuracy guess. A separate numerical tool will fit your chosen
model to your chosen data and produce the final accuracy forecast.

The training pool contains only valid labeled experiments on the same benchmark
and base-model family. Every pool item has an opaque alias, a session alias,
an exact151-feature vector, observed official target accuracy and its delta from
the valid parent/reference. The query has features but NO target outcome.
Aliases are only handles for selection and session-diversity checks, never
predictive features. feature_order specifies the vector interpretation.

Choose 8 to16 UNIQUE training examples spanning at least4 independent sessions.
Choose relevant and informative examples, including relevant bad/zero/degrading
outcomes. Do not cherry-pick successful examples or desired outcomes. Think about
the benchmark, parent strength, intervention type, training dose, context length,
data settings and the CHANGE relative to ancestor configurations. Similar-looking
recipes may fail; sensible intent does not guarantee positive delta. Prefer
coverage/support to fitting many parameters from very few observations.

Select one model from this fixed menu:
- weighted_median: predict the weighted median of selected observed deltas.
- ridge_10 or ridge_100: weighted linear delta regression after TRAIN-only
  standardization of selected columns; L2 alpha10 or100, with an intercept.
- shallow_tree: weighted delta decision tree, max_depth2,min_samples_leaf3.
Each predicted delta is added to the known query parent accuracy and final
accuracy is clipped to[0,1]. No LLM will override the tool's numeric forecast.

Supply a positive relevance weight in[0.25,4] per chosen example. The fitter
divides each by the number of selected examples from its session, then normalizes
all weights to mean1. This prevents repeated cards automatically dominating.
For a learned model select 1 to8 exact feature_order names, including
parent.accuracy. For weighted_median, feature_keys may be empty. Choose a few
meaningful settings/interactions available in the encoded columns, not IDs.

parent.accuracy is a real observed score or a published fixed-base reference,
distinguished by reference indicators. Zero is valid, not missing. A configuration
with observed=0 is UNKNOWN, not a known zero setting. History is summarized
pre-proposal configuration history, not additional outcome observations. Missing
data counts do not imply full-data training. Planned configurations do not prove
successful execution. You may use only the supplied training outcomes and your
prior knowledge; you cannot query the held-out outcome or change the feature set.

Return exactly one JSON object with these five keys:
selected_aliases: array of8-16 unique training aliases;
weights: array of equally many numbers in[0.25,4], in the same order;
model: one of the four exact model names;
feature_keys: array of at most8 distinct exact feature_order names;
rationale: at most100 words explaining relevance, model choice and uncertainty.
No other keys, markdown, model fits or numeric query forecast.
"""
PREDICT_SYSTEM = previous.SYSTEM + """
Each demonstration also has effective_fit_weight. These are the numerical model's
training weights after session adjustment; they can guide your calibration but
do not change the factual labels. You see the same selected training examples as
the fitted model, but not the selector rationale, its chosen model, fitted
coefficients, fitted predictions, or the full original training pool. Use all
supplied feature context to make your own forecast. You have no tools.
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def aliases_for(rows):
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("Only training examples may enter the candidate pool")
    if len({r["example_id"] for r in rows}) != len(rows) or len({r["benchmark"] for r in rows}) != 1:
        raise ValueError("Candidate identity or benchmark mismatch")
    def rank(value):
        return digest(SEED + "|" + value)
    ordered = sorted(rows, key=lambda r: rank(r["example_id"]))
    sessions = {cell: f"S{i:03d}" for i, cell in enumerate(sorted({r["cell_id"] for r in rows}, key=rank), 1)}
    return {f"T{i:03d}": {"example_id": r["example_id"], "session_alias": sessions[r["cell_id"]]} for i, r in enumerate(ordered, 1)}


def selector_prompt(query, pool, labels, aliases, order):
    validate_rows(pool, labels)
    validate_rows([query])
    if any(r["split"] != "train" for r in pool):
        raise ValueError("Only training examples may enter selector prompts")
    if aliases != aliases_for(pool):
        raise ValueError("Selector aliases differ from the complete training pool")
    if query["benchmark"] not in previous.TASKS or any(r["benchmark"] != query["benchmark"] for r in pool):
        raise ValueError("Cross-benchmark selection")
    if query["cell_id"] in {r["cell_id"] for r in pool}:
        raise ValueError("Query session appears in training pool")
    by_id = {r["example_id"]: r for r in pool}
    return canonical({
        "task": previous.TASKS[query["benchmark"]], "feature_order": order,
        "training_pool": [{
            "alias": alias, "session_alias": info["session_alias"],
            "features": previous.safe_vector(by_id[info["example_id"]], order),
            "official_accuracy": labels[info["example_id"]]["accuracy"],
            "official_delta": labels[info["example_id"]]["delta_accuracy"],
        } for alias, info in aliases.items()],
        "query_features": previous.safe_vector(query, order),
    })


def resolve_selection(response, aliases, pool, query):
    from tools.outcome_prediction.wm_agent_local_fit import validate_selection

    required = {"selected_aliases", "weights", "model", "feature_keys", "rationale"}
    if not isinstance(response, dict) or set(response) != required:
        raise ValueError("Selector must return the exact selection schema")
    choices = response["selected_aliases"]
    if not isinstance(choices, list) or any(not isinstance(a, str) or a not in aliases for a in choices):
        raise ValueError("Unknown or invalid training alias")
    selected = {"selected_ids": [aliases[a]["example_id"] for a in choices], **{k: response[k] for k in required - {"selected_aliases"}}}
    return validate_selection(selected, pool, query)


def selected_prompt(query, rows, labels, weights, order):
    validate_rows(rows, labels)
    validate_rows([query])
    if len(weights) != len(rows) or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Invalid effective demonstration weights")
    if query["cell_id"] in {r["cell_id"] for r in rows} or any(r["split"] != "train" or r["benchmark"] != query["benchmark"] for r in rows):
        raise ValueError("Unsafe selected demonstrations")
    return canonical({
        "task": previous.TASKS[query["benchmark"]], "feature_order": order,
        "labeled_training_demonstrations": [{
            "features": previous.safe_vector(r, order),
            "official_accuracy": labels[r["example_id"]]["accuracy"],
            "official_delta": labels[r["example_id"]]["delta_accuracy"],
            "effective_fit_weight": w,
        } for r, w in zip(rows, weights)],
        "target_features": previous.safe_vector(query, order),
    })


def prepare(training=TRAINING, previous_results=PREVIOUS, output=BUNDLE, regressors=REGRESSORS):
    training, previous_results, output, regressors = map(Path, (training, previous_results, output, regressors))
    if output.exists():
        raise FileExistsError("Choose a new immutable experiment directory")
    validate_bundle(training)
    previous.verify_files(previous_results, read(previous_results / "manifest.json"))
    previous.verify_files(regressors, read(regressors / "manifest.json"))
    rows, labels = read(training / "inputs.json"), read(training / "labels.json")
    validate_rows(rows, labels)
    train = [r for r in rows if r["split"] == "train"]
    queries = sorted([r for r in rows if r["split"] == "test"], key=lambda r: r["example_id"])
    if len(train) != 123 or len(queries) != 62:
        raise ValueError("Expected unchanged clean 123/62 split")
    train_labels = {r["example_id"]: labels[r["example_id"]] for r in train}
    order = previous.schema_keys()
    banks = {b: aliases_for([r for r in train if r["benchmark"] == b]) for b in previous.TASKS}
    packets = []
    for query in queries:
        pool = [r for r in train if r["benchmark"] == query["benchmark"]]
        local_labels = {r["example_id"]: train_labels[r["example_id"]] for r in pool}
        prompt = selector_prompt(query, pool, local_labels, banks[query["benchmark"]], order)
        packets.append({"example_id": query["example_id"], "benchmark": query["benchmark"], "prompt": prompt, "prompt_sha256": digest(prompt)})
    policy = {
        "schema": "agent-selected-local-model-v1", "created_at": now(),
        "training": str(training.resolve()), "training_manifest_sha256": sha(training / "manifest.json"),
        "previous_results": str(previous_results.resolve()), "previous_manifest_sha256": sha(previous_results / "manifest.json"),
        "regressors": str(regressors.resolve()), "regressor_manifest_sha256": sha(regressors / "manifest.json"),
        "libraries": {name: version(name) for name in ("numpy", "scikit-learn")},
        "sources": {str((Path(__file__).parent / name).resolve()): sha(Path(__file__).parent / name) for name in ("wm_agent_local_benchmark.py", "wm_agent_local_fit.py", "wm_local_agent_io.py", "wm_fresh_llm_baseline.py")},
        "model_requested": "claude-opus-5", "effort": "high", "timeout_seconds": 240,
        "max_cli_network_retries": 1, "max_attempts": 2, "workers": 4,
        "selector_max_output_tokens": 4096, "selector_budget_usd": 0.85,
        "predictor_max_output_tokens": 2048, "predictor_budget_usd": 0.50,
        "reported_cost_stop_usd": 50.0,
        "selector_system": SELECT_SYSTEM, "predictor_system": PREDICT_SYSTEM,
        "feature_order": order, "model_menu": MODELS, "seed": SEED,
        "primary": "agent_local_model", "comparators": ["selected_few_shot", "fixed_few_shot", "selected_weighted_median"],
        "selection": "One valid agent choice per test query from all same-benchmark TRAIN examples; no fit or target-error feedback. Fixed menu/constraints before calls; no held-out tuning.",
        "features": "Unchanged151 numeric features. Agent chooses<=8 existing columns for learned model; direct LLM sees151 on identical selected examples and effective weights, no selector rationale/model/forecast.",
        "fit": "Only valid known target/reference TRAIN rows; 8-16 selected examples>=4sessions. Preserve failures/zeros. Numeric model prediction is final; no LLM override.",
        "weights": "agent relevance / selected cards in session, normalized to mean1; raw relevance in[0.25,4]",
        "validation": "Conditional selected-subset leave-one-session-out diagnostics are NOT unbiased validation of the selection policy, which has seen those labels. This predeclared pilot has no model-policy hyperparameter search.",
        "metric": "Primary equal-session MAE per benchmark; same common successful testIDs for paired arms; report coverage, all attempts, pooledMAE, accuracy/deltaR2, paired session bootstrap.",
        "limits": ["Previously explored developmental holdout; current rationale analysis informed this hypothesis, not confirmatory.", "Same151 isolates selection/modelstrategy, not the newly discovered omitted raw recipe features.", "Selector sees all training labels, old fixedfewshot sees16. Shared-selector cost applies to both agent-selected arms.", "Within selected arms, local model uses<=8 columns while LLM may use151. Labels/rows/weights identical.", "CLI reported costs are equivalents, not actual subscription billing; stop threshold checked between calls with bounded concurrency."],
    }
    output.mkdir(parents=True, mode=0o700)
    for name, value in (("train_inputs.json", train), ("train_labels.json", train_labels), ("queries.json", queries), ("aliases.json", banks), ("selector_prompts.json", packets), ("policy.json", policy)):
        write(output / name, value)
    write(output / "manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    validate_prepared(output)
    return {"train": len(train), "queries": len(queries), "selector_prompt_chars": [min(len(p["prompt"]) for p in packets), max(len(p["prompt"]) for p in packets)]}


def validate_prepared(bundle):
    bundle = Path(bundle)
    manifest = read(bundle / "manifest.json")
    required = {"train_inputs.json", "train_labels.json", "queries.json", "aliases.json", "selector_prompts.json", "policy.json"}
    if set(manifest) != required:
        raise ValueError("Incomplete experiment manifest")
    previous.verify_files(bundle, manifest)
    policy = read(bundle / "policy.json")
    for path, expected in policy["sources"].items():
        if sha(path) != expected:
            raise ValueError("Changed frozen source: " + path)
    if policy["selector_system"] != SELECT_SYSTEM or policy["predictor_system"] != PREDICT_SYSTEM or policy["model_menu"] != list(MODELS):
        raise ValueError("Changed agent specification")
    training, prior = Path(policy["training"]), Path(policy["previous_results"])
    if sha(training / "manifest.json") != policy["training_manifest_sha256"] or sha(prior / "manifest.json") != policy["previous_manifest_sha256"]:
        raise ValueError("Changed upstream manifest")
    validate_bundle(training)
    previous.verify_files(prior, read(prior / "manifest.json"))
    regressors = Path(policy["regressors"])
    if sha(regressors / "manifest.json") != policy["regressor_manifest_sha256"]:
        raise ValueError("Changed regression manifest")
    previous.verify_files(regressors, read(regressors / "manifest.json"))
    if policy["libraries"] != {name: version(name) for name in policy["libraries"]}:
        raise ValueError("Changed numerical library versions")
    source_rows, source_labels = read(training / "inputs.json"), read(training / "labels.json")
    train, queries, labels = (read(bundle / name) for name in ("train_inputs.json", "queries.json", "train_labels.json"))
    if train != [r for r in source_rows if r["split"] == "train"] or queries != sorted([r for r in source_rows if r["split"] == "test"], key=lambda r: r["example_id"]):
        raise ValueError("Changed train/query rows")
    if labels != {r["example_id"]: source_labels[r["example_id"]] for r in train}:
        raise ValueError("Changed training labels")
    validate_rows(train, labels)
    validate_rows(queries)
    banks = read(bundle / "aliases.json")
    expected_banks = {b: aliases_for([r for r in train if r["benchmark"] == b]) for b in previous.TASKS}
    if banks != expected_banks:
        raise ValueError("Changed training aliases")
    packets = read(bundle / "selector_prompts.json")
    if len(packets) != len(queries) or len({p["example_id"] for p in packets}) != len(queries):
        raise ValueError("Duplicate or missing selector packets")
    for packet, query in zip(packets, queries):
        pool = [r for r in train if r["benchmark"] == query["benchmark"]]
        prompt = selector_prompt(query, pool, {r["example_id"]: labels[r["example_id"]] for r in pool}, banks[query["benchmark"]], policy["feature_order"])
        if packet != {"example_id": query["example_id"], "benchmark": query["benchmark"], "prompt": prompt, "prompt_sha256": digest(prompt)}:
            raise ValueError("Selector packet differs from source-derived TRAIN-only prompt")
    return policy, train, labels, queries, banks, packets


def key_for(query):
    return query["example_id"].replace("/", "__")


def validate_success_call(record, policy):
    from tools.outcome_prediction.wm_local_agent_io import _decode, _events

    events = _events(record["stdout"])
    response, error, models, _ = _decode(events, policy["model_requested"], record["returncode"], record["stderr"])
    if error or events != record["raw_events"] or response != record["response"] or models != record["models_resolved"] or record["model_requested"] != policy["model_requested"] or record["effort"] != policy["effort"]:
        raise ValueError("Successful call fails raw tool-free/model/response verification")


def compare_fixed_holdout(rows, labels, prediction, reference):
    result = paired_comparison(rows, labels, prediction, reference)
    result["interpretation"] = "Exploratory paired bootstrap of fixed held-out session errors; this previously explored developmental holdout is not independent confirmation or OOF selection-policy evaluation."
    return result


def run(bundle=BUNDLE, output=OUTPUT):
    from tools.outcome_prediction.wm_agent_local_fit import fit_predict
    from tools.outcome_prediction.wm_local_agent_io import call_json

    bundle, output = Path(bundle), Path(output)
    policy, train, labels, queries, banks, packets = validate_prepared(bundle)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name in ("calls", "selected_prompts", "query_results"):
        (output / name).mkdir(mode=0o700)
    write(output / "run_policy.json", {"started_at": now(), "bundle": str(bundle.resolve()), "bundle_manifest_sha256": sha(bundle / "manifest.json"), "claude_version": subprocess.run(["claude", "--version"], capture_output=True, text=True, check=True).stdout.strip()})
    stop = threading.Event()
    lock = threading.Lock()
    total_cost = 0.0
    started = time.perf_counter()

    def invoke(prompt, stage, system=None):
        nonlocal total_cost
        if stop.is_set():
            return {"response": None, "error": "unattempted_after_systemic_stop", "created_at": now(), "attempted": False}
        stage_policy = {**policy, "system": system or policy[stage + "_system"], "max_output_tokens": policy[stage + "_max_output_tokens"], "per_call_budget_usd": policy[stage + "_budget_usd"]}
        try:
            result = call_json(prompt, stage_policy)
            result["attempted"] = True
        except subprocess.TimeoutExpired:
            result = {"response": None, "error": "request_timeout", "created_at": now(), "attempted": True}
        with lock:
            total_cost += result.get("cost_usd_reported") or 0.0
            if total_cost >= policy["reported_cost_stop_usd"]:
                stop.set()
        if any(term in str(result.get("error")).lower() for term in ("rate limit", "rate_limit", "usage limit", "session limit", "hit your limit", "authentication", "model identity", "unexpected tools", "billing", "credit balance")):
            stop.set()
        return result

    probe = invoke('Return exactly {"ok":true}. No tools.', "selector", system="This is a connectivity check. Return exactly the requested JSON object. You have no tools.")
    write(output / "probe.json", probe)
    if probe.get("error") or probe.get("response") != {"ok": True}:
        raise RuntimeError("Isolation/connectivity probe failed")
    packet_of = {p["example_id"]: p for p in packets}

    def work(query):
        packet = packet_of[query["example_id"]]
        key = key_for(query)
        pool = [r for r in train if r["benchmark"] == query["benchmark"]]
        selection = None
        result = {"example_id": query["example_id"], "benchmark": query["benchmark"], "selector_prompt_sha256": packet["prompt_sha256"], "selection": None, "local_fit": None, "median_fit": None, "llm_prediction": None, "error": None}
        for attempt in range(1, policy["max_attempts"] + 1):
            record = invoke(packet["prompt"], "selector")
            if record.get("response") is not None and not record.get("error"):
                try:
                    selection = resolve_selection(record["response"], banks[query["benchmark"]], pool, query)
                except (ValueError, TypeError, KeyError) as exc:
                    record["error"] = "invalid_selection: " + str(exc)
            record.update(example_id=query["example_id"], stage="selector", attempt=attempt, prompt_sha256=packet["prompt_sha256"])
            write(output / "calls" / (key + f"__selector__attempt{attempt}.json"), record)
            result["selector_attempts"] = attempt
            if selection is not None or stop.is_set():
                break
        result["selection"] = selection
        if selection is None:
            result["error"] = record["error"] or "no_valid_selection"
            write(output / "query_results" / (key + ".json"), result)
            return result
        selected = [{r["example_id"]: r for r in pool}[k] for k in selection["selected_ids"]]
        local_labels = {r["example_id"]: labels[r["example_id"]] for r in selected}
        try:
            local = fit_predict(selected, local_labels, query, selection)
            median_spec = {**selection, "model": "weighted_median", "feature_keys": []}
            median = fit_predict(selected, local_labels, query, median_spec)
            result.update(local_fit=local, median_fit=median)
            weights = local["fit_metadata"]["normalized_sample_weights"]
            prompt = selected_prompt(query, selected, local_labels, weights, policy["feature_order"])
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            result["error"] = "fit_error: " + str(exc)
            write(output / "query_results" / (key + ".json"), result)
            return result
        prompt_hash = digest(prompt)
        write(output / "selected_prompts" / (key + ".json"), {"example_id": query["example_id"], "prompt": prompt, "prompt_sha256": prompt_hash, "selected_ids": selection["selected_ids"], "effective_weights": weights})
        for attempt in range(1, policy["max_attempts"] + 1):
            record = invoke(prompt, "predictor")
            if record.get("response") is not None and not record.get("error"):
                try:
                    result["llm_prediction"] = previous.parse_prediction(canonical(record["response"]))
                except (ValueError, TypeError, KeyError) as exc:
                    record["error"] = "invalid_forecast: " + str(exc)
            record.update(example_id=query["example_id"], stage="predictor", attempt=attempt, prompt_sha256=prompt_hash)
            write(output / "calls" / (key + f"__predictor__attempt{attempt}.json"), record)
            result["predictor_attempts"] = attempt
            if result["llm_prediction"] is not None or stop.is_set():
                break
        result["error"] = None if result["llm_prediction"] is not None else record["error"]
        write(output / "query_results" / (key + ".json"), result)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=policy["workers"]) as executor:
        pending = [executor.submit(work, query) for query in queries]
        for future in as_completed(pending):
            row = future.result()
            results.append(row)
            model = (row["selection"] or {}).get("model")
            print(f"{len(results)}/{len(queries)} {row['example_id']} model={model} status={row['error'] or 'ok'} reported_total={total_cost:.2f}", flush=True)
    results.sort(key=lambda r: r["example_id"])
    write(output / "predictions.json", results)
    write(output / "prediction_freeze.json", {"created_at": now(), "elapsed_seconds": time.perf_counter() - started, "reported_cost_usd": total_cost, "systemic_stop": stop.is_set(), "files": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}, "note": "All selections, fits and forecasts saved before target scoring. No test outcomes entered any selection or fit."})
    return {"queries": len(results), "local_success": sum(r["local_fit"] is not None for r in results), "llm_success": sum(r["llm_prediction"] is not None for r in results), "reported_cost_usd": total_cost}


def score(bundle=BUNDLE, output=OUTPUT):
    from tools.outcome_prediction.wm_agent_local_fit import fit_predict

    bundle, output = Path(bundle), Path(output)
    if (output / "report.json").exists() or (output / "manifest.json").exists():
        raise FileExistsError("Scoring outputs are immutable; audit existing report separately")
    policy, train, train_labels, queries, banks, packets = validate_prepared(bundle)
    freeze = read(output / "prediction_freeze.json")
    previous.verify_files(output, freeze["files"])
    predictions = read(output / "predictions.json")
    by_id = {p["example_id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {q["example_id"] for q in queries}:
        raise ValueError("Missing/duplicate forecast identities")
    train_by_id = {r["example_id"]: r for r in train}
    packet_of = {p["example_id"]: p for p in packets}
    for query in queries:
        value = by_id[query["example_id"]]
        key = key_for(query)
        packet = packet_of[query["example_id"]]
        if value["selector_prompt_sha256"] != packet["prompt_sha256"] or value != read(output / "query_results" / (key + ".json")):
            raise ValueError("Query/prompt alignment mismatch")
        if value["selection"] is None:
            continue
        calls = [read(output / "calls" / (key + f"__selector__attempt{i}.json")) for i in range(1, value["selector_attempts"] + 1)]
        valid = [c for c in calls if not c.get("error")]
        if len(valid) != 1 or valid[0] is not calls[-1] or any(c["prompt_sha256"] != packet["prompt_sha256"] or c["example_id"] != query["example_id"] or c["stage"] != "selector" or c["attempt"] != i for i, c in enumerate(calls, 1)):
            raise ValueError("Not first valid selector response")
        validate_success_call(valid[0], policy)
        selection = resolve_selection(valid[0]["response"], banks[query["benchmark"]], [r for r in train if r["benchmark"] == query["benchmark"]], query)
        if selection != value["selection"]:
            raise ValueError("Saved selection differs from raw choice")
        if value["local_fit"] is None:
            continue
        chosen = [train_by_id[k] for k in selection["selected_ids"]]
        labels = {r["example_id"]: train_labels[r["example_id"]] for r in chosen}
        # Deterministic replay on TRAIN labels only verifies saved fits; no model
        # choice or hyperparameter changes are made using held-out outcomes.
        expected_fit = fit_predict(chosen, labels, query, selection)
        expected_median = fit_predict(chosen, labels, query, {**selection, "model": "weighted_median", "feature_keys": []})
        for actual, expected in ((value["local_fit"], expected_fit), (value["median_fit"], expected_median)):
            if actual != expected:
                raise ValueError("Saved local fit fails deterministic replay")
        prompt = selected_prompt(query, chosen, labels, expected_fit["fit_metadata"]["normalized_sample_weights"], policy["feature_order"])
        saved = read(output / "selected_prompts" / (key + ".json"))
        if saved != {"example_id": query["example_id"], "prompt": prompt, "prompt_sha256": digest(prompt), "selected_ids": selection["selected_ids"], "effective_weights": expected_fit["fit_metadata"]["normalized_sample_weights"]}:
            raise ValueError("Changed selected-example prompt")
        if value["llm_prediction"] is not None:
            calls = [read(output / "calls" / (key + f"__predictor__attempt{i}.json")) for i in range(1, value["predictor_attempts"] + 1)]
            valid = [c for c in calls if not c.get("error")]
            if len(valid) != 1 or valid[0] is not calls[-1] or any(c["prompt_sha256"] != digest(prompt) or c["example_id"] != query["example_id"] or c["stage"] != "predictor" or c["attempt"] != i for i, c in enumerate(calls, 1)):
                raise ValueError("Invalid selected-LLM response sequence")
            validate_success_call(valid[0], policy)
            if value["llm_prediction"] != previous.parse_prediction(canonical(valid[0]["response"])):
                raise ValueError("Changed selected-LLM forecast")
    labels = read(Path(policy["training"]) / "labels.json")
    fixed = {(r["example_id"], r["arm"]): r for r in read(Path(policy["previous_results"]) / "predictions.json")}
    reg_dir = Path(policy["regressors"])
    previous.verify_files(reg_dir, read(reg_dir / "manifest.json"))
    reg = read(reg_dir / "predictions.json")
    report = {"scored_at": now(), "prediction_frozen_at": freeze["created_at"], "benchmarks": {}, "reported_cost_usd": freeze["reported_cost_usd"], "primary": policy["primary"], "note": "Same common successful IDs per benchmark; selection was conditioned onTRAIN labels only. Paired bootstrap is of fixed development-holdout session errors, not OOF."}
    for benchmark in previous.TASKS:
        intended = [q for q in queries if q["benchmark"] == benchmark]
        common = [q for q in intended if by_id[q["example_id"]]["local_fit"] is not None and by_id[q["example_id"]]["llm_prediction"] is not None]
        ids = [q["example_id"] for q in common]
        y = np.asarray([labels[k]["accuracy"] for k in ids])
        reference = np.asarray([q["parent_reference"] for q in common])
        forecasts = {
            "agent_local_model": np.asarray([by_id[k]["local_fit"]["predicted_accuracy"] for k in ids]),
            "selected_weighted_median": np.asarray([by_id[k]["median_fit"]["predicted_accuracy"] for k in ids]),
            "selected_few_shot": np.asarray([by_id[k]["llm_prediction"]["predicted_accuracy"] for k in ids]),
            "fixed_few_shot": np.asarray([fixed[k, "few_shot"]["predicted_accuracy"] for k in ids]),
            "fixed_zero_shot": np.asarray([fixed[k, "zero_shot"]["predicted_accuracy"] for k in ids]),
            **{m: np.asarray([reg["combined__" + benchmark][k][m] for k in ids]) for m in ("train_selected", "hgb_current", "hgb_history")},
        }
        metrics = {}
        for arm, pred in forecasts.items():
            metrics[arm] = {}
            for subset in ("all", "published_base", "measured_parent"):
                mask = np.asarray([subset == "all" or q["reference_kind"] == subset for q in common], dtype=bool)
                subrows = [q for q, keep in zip(common, mask) if keep]
                metrics[arm][subset] = {"accuracy": evaluate(subrows, y[mask], pred[mask]), "delta": evaluate(subrows, (y - reference)[mask], (pred - reference)[mask])}
        report["benchmarks"][benchmark] = {
            "intended": len(intended), "common_success": len(common), "common_ids": ids,
            "coverage": {"local_fit": sum(by_id[q["example_id"]]["local_fit"] is not None for q in intended), "selected_llm": sum(by_id[q["example_id"]]["llm_prediction"] is not None for q in intended)},
            "model_choices": dict(Counter(by_id[k]["selection"]["model"] for k in ids)),
            "metrics": metrics,
            "primary_comparisons": {arm: compare_fixed_holdout(common, y, forecasts["agent_local_model"], forecasts[arm]) for arm in ("fixed_few_shot", "selected_few_shot", "selected_weighted_median")} if common else {},
        }
    write(output / "report.json", report)
    write(output / "manifest.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == "prepare":
        value = prepare(output=args.bundle)
    elif args.action == "run":
        value = run(args.bundle, args.output)
    else:
        report = score(args.bundle, args.output)
        value = {b: {"coverage": r["common_success"], "mae": {k: v["all"]["accuracy"]["mae"] for k, v in r["metrics"].items()}} for b, r in report["benchmarks"].items()}
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
