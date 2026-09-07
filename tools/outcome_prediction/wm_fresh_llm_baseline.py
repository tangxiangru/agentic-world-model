"""Fresh, tool-free Claude scalar forecasts on the frozen clean delta split.

Only NEW additive artifacts are written. Prompts expose exactly the frozen
151-column numeric history view, never unrestricted trajectory text. No model
fits occur. Few-shot examples come exclusively from current training sessions.
All forecasts must be frozen before scoring against held-out labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tools.outcome_prediction.wm_clean_refresh import read, sha, write
from tools.outcome_prediction.wm_one_step_features import features
from tools.outcome_prediction.wm_one_step_train import validate_bundle, validate_rows
from tools.outcome_prediction.wm_small_benchmark import bootstrap_gain, evaluate

DEFAULT_TRAINING = Path("data/analysis/wm_one_step_predictors/v1_bundle")
DEFAULT_REGRESSORS = Path("data/analysis/wm_one_step_predictors/v1_results")
DEFAULT_BUNDLE = Path("data/analysis/wm_fresh_llm/v1_bundle")
DEFAULT_OUTPUT = Path("data/analysis/wm_fresh_llm/v1_results")
ARMS = ("zero_shot", "few_shot")
SEED = "fresh-llm-20260906-v1"
BANK_SIZE = 16
TASKS = {
    "gsm8k": {"base_model": "google/gemma-3-4b-pt", "benchmark": "GSM8K", "test_problems": 1319},
    "aime2025": {"base_model": "Qwen/Qwen3-4B-Base", "benchmark": "AIME 2025", "test_problems": 30},
}
SYSTEM = """You forecast post-training experiment outcomes for small language models.
Predict the target checkpoint's official exact-match test accuracy from the supplied
immediate-parent/reference accuracy and prospective recipe/code/history features.
This is a scalar prediction task, not an agentic experiment or a search task. You
have no tools. Do not assume additional files, experiment results or external data.

Inputs are fixed numeric summaries of planned configurations and statically parsed
pre-proposal code. They do NOT certify that training succeeded. Accuracy may improve,
stay unchanged, deteriorate, or be zero. Reason about training dose, data, intervention
type, missing information, and uncertainty. If labeled demonstrations are supplied,
use them to calibrate your forecast; otherwise rely on the task and model knowledge.

Feature arrays follow the shared feature_order exactly. parent.accuracy is a valid
observed checkpoint score or a published fixed-base reference, distinguished by the
two reference indicators. A real zero accuracy is valid. Current observed masks tell
whether each configuration is known: if observed=0 the accompanying zero value is
MISSING, not a zero setting. Family fields are multi-hot. History is oldest-to-newest
recipe history represented by fixed summaries: means/maxima, the immediate parent's
latest values, and recency-weighted means with a one-step half-life. Observed fractions
and latest_observed masks distinguish missing values; no ancestor outcome scores are
provided. History settings summarize declarations, not achieved training exposure.
History steps=0 means there are no checkpoint-recipe ancestors (a fixed-base start).
Quarantined/unavailable history content is absent and reflected by coverage indicators.

Give one point forecast aimed at low absolute error, not an optimistic best case.
Return exactly one JSON object with predicted_accuracy (a finite number in [0,1])
and rationale (at most 60 words). Do not give percentages or any other output.
The evaluator will compute predicted_delta = predicted_accuracy - parent.accuracy.
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rank_key(value):
    return digest_text(SEED + "|" + value)


def schema_keys():
    return sorted(features({
        "parent": {"accuracy": 0.0, "kind": "published_base", "reference_known": True},
        "model_input": {}, "history": [], "history_complete_to_base": True,
    }, "history"))


def safe_vector(row, order):
    view = row["views"]["history"]
    if sorted(view) != order or len(order) != 151:
        raise ValueError("Unexpected feature schema")
    result = [view[k] for k in order]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in result):
        raise ValueError("Only finite numeric whitelist values may enter prompts")
    if view["parent.accuracy"] != row["parent_reference"]:
        raise ValueError("Parent reference mismatch")
    return result


def select_bank(train, benchmark, size=BANK_SIZE):
    """Hash-ordered session round robin; no label argument or target similarity."""
    grouped = defaultdict(list)
    for row in train:
        if row["split"] != "train":
            raise ValueError("Only training rows may enter demonstration selection")
        if row["benchmark"] == benchmark:
            grouped[row["cell_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda r: rank_key(r["example_id"]))
    sessions = sorted(grouped, key=rank_key)
    selected, depth = [], 0
    while len(selected) < size:
        added = False
        for cell in sessions:
            if depth < len(grouped[cell]):
                selected.append(grouped[cell][depth])
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
        depth += 1
    return selected


def prompt_for(row, bank, train_labels, order):
    safe = {
        "task": TASKS[row["benchmark"]],
        "feature_order": order,
        "labeled_training_demonstrations": [
            {"features": safe_vector(b, order),
             "official_accuracy": train_labels[b["example_id"]]["accuracy"],
             "official_delta": train_labels[b["example_id"]]["delta_accuracy"]}
            for b in bank
        ],
        "target_features": safe_vector(row, order),
    }
    return canonical(safe)


def verify_files(directory, manifest):
    directory = Path(directory).resolve()
    for name, expected in manifest.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != expected:
            raise ValueError("Changed frozen artifact: " + name)


def prepare(training=DEFAULT_TRAINING, regressors=DEFAULT_REGRESSORS, output=DEFAULT_BUNDLE):
    training, regressors, output = map(Path, (training, regressors, output))
    if output.exists():
        raise FileExistsError("Never overwrite a frozen prompt bundle")
    validate_bundle(training)
    verify_files(regressors, read(regressors / "manifest.json"))
    rows, labels = read(training / "inputs.json"), read(training / "labels.json")
    validate_rows(rows, labels)
    train = [r for r in rows if r["split"] == "train"]
    test = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["example_id"])
    if len(train) != 123 or len(test) != 62:
        raise ValueError("Expected unchanged 123/62 complete-delta split")
    if {r["cell_id"] for r in train} & {r["cell_id"] for r in test}:
        raise ValueError("Training/test session overlap")
    train_labels = {r["example_id"]: labels[r["example_id"]] for r in train}
    order = schema_keys()
    banks = {b: select_bank(train, b) for b in TASKS}
    prompts = []
    for row in test:
        for arm in ARMS:
            bank = banks[row["benchmark"]] if arm == "few_shot" else []
            prompt = prompt_for(row, bank, train_labels, order)
            prompts.append({
                "example_id": row["example_id"], "cell_id": row["cell_id"],
                "benchmark": row["benchmark"], "arm": arm,
                "parent_reference": row["parent_reference"],
                "reference_kind": row["reference_kind"],
                "prompt": prompt, "prompt_sha256": digest_text(prompt),
                "bank_ids": [r["example_id"] for r in bank],
                "bank_sessions": [r["cell_id"] for r in bank],
            })
    policy = {
        "schema": "fresh-matched-feature-scalar-llm-v1", "created_at": now(),
        "model_requested": "claude-opus-5", "effort": "high",
        "max_output_tokens": 2048, "per_call_budget_usd": 0.50,
        "timeout_seconds": 240, "workers": 4, "max_attempts": 2,
        "max_cli_network_retries": 1, "arms": ARMS,
        "primary_llm_arm": "few_shot", "primary_regressor_arm": "train_selected",
        "system": SYSTEM, "system_sha256": digest_text(SYSTEM),
        "training": str(training.resolve()), "regressors": str(regressors.resolve()),
        "training_manifest_sha256": sha(training / "manifest.json"),
        "regressor_manifest_sha256": sha(regressors / "manifest.json"),
        "feature_order": order, "bank_size": BANK_SIZE, "seed": SEED,
        "bank_selection": "Same-benchmark TRAIN-only SHA256(seed|session), then SHA256(seed|example), session round robin; independent of labels and test features.",
        "input_contract": "Exact frozen history151 numeric view; parent/current views are subsets. No IDs, raw text/code, target outcomes or ancestor outcomes sent to model.",
        "clean_only": "Every row has known valid target and parent/reference accuracy; missing rows never fit, demonstrate or test. Real zero retained.",
        "limitations": [
            "Matched-feature scalar LLM baseline, not full recipe/code or RPM subtree-ranking replication.",
            "Regressors see all matching-benchmark training rows; few-shot sees deterministic16. Zero-shot sees none.",
            "Previously explored development holdout; no prompt tuning or arm selection on new test scores.",
            "Published base references are shared constants, not per-run paired measurements.",
            "CLI default sampling is uncontrolled; single forecast per arm/example, no best-of selection.",
        ],
        "sources": {str(Path(__file__).resolve()): sha(__file__)},
        "provider_documents": ["https://platform.claude.com/docs/en/models/overview",
                               "https://code.claude.com/docs/en/env-vars"],
    }
    output.mkdir(parents=True, mode=0o700)
    for name, value in (("policy.json", policy), ("prompts.json", prompts),
                        ("bank_manifest.json", {b: {"ids": [r["example_id"] for r in bank], "sessions": [r["cell_id"] for r in bank]} for b, bank in banks.items()})):
        write(output / name, value)
    write(output / "manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    validate_prompt_bundle(output)
    return {"prompts": len(prompts), "test_rows": len(test),
            "prompt_char_range": [min(len(p["prompt"]) for p in prompts), max(len(p["prompt"]) for p in prompts)]}


def validate_prompt_bundle(bundle):
    bundle = Path(bundle)
    verify_files(bundle, read(bundle / "manifest.json"))
    policy, prompts = read(bundle / "policy.json"), read(bundle / "prompts.json")
    if policy["system"] != SYSTEM or policy["system_sha256"] != digest_text(SYSTEM):
        raise ValueError("Changed system prompt")
    for path, expected in policy["sources"].items():
        if sha(path) != expected:
            raise ValueError("Changed frozen inference source")
    training = Path(policy["training"])
    if sha(training / "manifest.json") != policy["training_manifest_sha256"]:
        raise ValueError("Changed frozen training bundle")
    validate_bundle(training)
    rows, labels = read(training / "inputs.json"), read(training / "labels.json")
    validate_rows(rows, labels)
    train = [r for r in rows if r["split"] == "train"]
    test = {r["example_id"]: r for r in rows if r["split"] == "test"}
    train_labels = {r["example_id"]: labels[r["example_id"]] for r in train}
    banks = {b: select_bank(train, b) for b in TASKS}
    identities = [(p["example_id"], p["arm"]) for p in prompts]
    if len(set(identities)) != len(identities) or set(identities) != {(k, a) for k in test for a in ARMS}:
        raise ValueError("Prompt coverage mismatch")
    if policy["feature_order"] != schema_keys():
        raise ValueError("Unknown feature names")
    for packet in prompts:
        row = test[packet["example_id"]]
        bank = banks[row["benchmark"]] if packet["arm"] == "few_shot" else []
        expected = prompt_for(row, bank, train_labels, policy["feature_order"])
        if (packet["prompt"] != expected or packet["prompt_sha256"] != digest_text(expected)
                or packet["bank_ids"] != [b["example_id"] for b in bank]
                or packet["bank_sessions"] != [b["cell_id"] for b in bank]
                or any(packet[k] != row[k] for k in ("cell_id", "benchmark", "parent_reference", "reference_kind"))):
            raise ValueError("Prompt content or train-only bank mismatch")
    return policy, prompts


def parse_prediction(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise TypeError("Forecast must be one JSON object")
    number = data.get("predicted_accuracy")
    if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Invalid forecast; no missing/out-of-range imputation")
    if not isinstance(data.get("rationale"), str):
        raise TypeError("Missing rationale")
    return {"predicted_accuracy": float(number), "rationale": data["rationale"]}


def call_model(prompt, policy, *, system=None):
    """No tools, no repo cwd, no prompt references to local files; no shell."""
    command = ["claude", "-p", "--safe-mode", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
               "--no-session-persistence", "--disable-slash-commands", "--setting-sources", "",
               "--tools", "", "--output-format", "stream-json", "--verbose",
               "--effort", policy["effort"], "--max-budget-usd", str(policy["per_call_budget_usd"]),
               "--model", policy["model_requested"], "--system-prompt", system or policy["system"]]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_") and k != "CLAUDECODE"}
    env.update(CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(policy["max_output_tokens"]),
               CLAUDE_CODE_MAX_RETRIES=str(policy["max_cli_network_retries"]))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="wm-llm-no-data-") as cwd:
        proc = subprocess.run(command, input=prompt, text=True, capture_output=True,
                              cwd=cwd, env=env, timeout=policy["timeout_seconds"], check=False)
    events = []
    for line in proc.stdout.splitlines():
        if line.strip():
            try:
                event = json.loads(line)
                events.append(event if isinstance(event, dict) else {"unexpected_json": event})
            except json.JSONDecodeError:
                events.append({"unparsed_stdout": line})
    init = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    results = [e for e in events if e.get("type") == "result"]
    result = results[-1] if results else {}
    assistants = [e.get("message", {}) for e in events if e.get("type") == "assistant"]
    models = sorted({m["model"] for m in assistants if m.get("model")})
    tool_uses = [c for m in assistants for c in m.get("content", []) if c.get("type") == "tool_use"]
    error = None
    prediction = None
    try:
        if proc.returncode or result.get("is_error") or not results:
            raise ValueError("CLI request failed: " + str(result.get("result", proc.stderr[:300])))
        if len(init) != 1 or init[0].get("tools") or init[0].get("mcp_servers") or tool_uses:
            raise ValueError("Missing tool-free initialization or unexpected tools")
        if init[0].get("model") != policy["model_requested"] or models != [policy["model_requested"]]:
            raise ValueError("Requested/resolved model identity mismatch: " + str(models))
        prediction = parse_prediction(result.get("result", ""))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        error = str(exc)
    return {
        "created_at": now(), "elapsed_seconds": time.perf_counter() - started,
        "model_requested": policy["model_requested"], "models_resolved": models,
        "effort": policy["effort"], "prediction": prediction, "error": error,
        "cost_usd_reported": result.get("total_cost_usd"), "usage": result.get("usage"),
        "model_usage": result.get("modelUsage"), "returncode": proc.returncode,
        "raw_events": events, "stderr": proc.stderr,
    }


def prediction_key(packet):
    return packet["arm"] + "__" + packet["example_id"].replace("/", "__")


def run(bundle=DEFAULT_BUNDLE, output=DEFAULT_OUTPUT):
    bundle, output = Path(bundle), Path(output)
    policy, prompts = validate_prompt_bundle(bundle)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    write(output / "run_policy.json", {"started_at": now(), "bundle": str(bundle.resolve()),
          "bundle_manifest_sha256": sha(bundle / "manifest.json"),
          "claude_version": subprocess.run(["claude", "--version"], capture_output=True, text=True, check=True).stdout.strip(),
          "note": "No test labels are passed to model calls. One no-data model/tool isolation probe precedes target forecasts."})
    probe = call_model('Return exactly {"predicted_accuracy":0.5,"rationale":"connectivity probe"}', policy,
                       system="Return the requested JSON object verbatim. No tools or other content.")
    write(output / "probe.json", probe)
    if probe["error"]:
        raise RuntimeError("Model isolation/connectivity probe failed: " + probe["error"])
    (output / "calls").mkdir(mode=0o700)
    systemic_stop = threading.Event()

    def work(packet):
        attempts = []
        for attempt in range(1, policy["max_attempts"] + 1):
            try:
                if systemic_stop.is_set():
                    record = {"prediction": None, "error": "unattempted_after_systemic_error",
                              "created_at": now(), "attempted": False}
                else:
                    record = call_model(packet["prompt"], policy)
                    record["attempted"] = True
            except subprocess.TimeoutExpired:
                record = {"prediction": None, "error": "request_timeout", "created_at": now()}
            record.update(example_id=packet["example_id"], arm=packet["arm"],
                          prompt_sha256=packet["prompt_sha256"], attempt=attempt)
            write(output / "calls" / (prediction_key(packet) + f"__attempt{attempt}.json"), record)
            attempts.append(record)
            if record["prediction"] is not None:
                break
            if any(term in str(record.get("error")).lower() for term in (
                "rate limit", "rate_limit", "session limit", "usage limit", "hit your limit",
                "limit reached", "model identity", "unexpected tools", "authentication",
                "billing", "credit balance", "unattempted_after_systemic_error",
            )):
                systemic_stop.set()
                break
        last = attempts[-1]
        value = None if last["prediction"] is None else last["prediction"]["predicted_accuracy"]
        return {"example_id": packet["example_id"], "arm": packet["arm"],
                "prompt_sha256": packet["prompt_sha256"], "attempts": len(attempts),
                "predicted_accuracy": value,
                "predicted_delta": None if value is None else value - packet["parent_reference"],
                "error": last["error"], "cost_usd_reported": sum(r.get("cost_usd_reported") or 0 for r in attempts)}

    predictions = []
    with ThreadPoolExecutor(max_workers=policy["workers"]) as pool:
        futures = [pool.submit(work, packet) for packet in prompts]
        for future in as_completed(futures):
            value = future.result()
            predictions.append(value)
            print(f"{len(predictions)}/{len(prompts)} {value['arm']} {value['example_id']} "
                  f"status={'ok' if value['predicted_accuracy'] is not None else 'FAILED'} "
                  f"cost={value['cost_usd_reported']:.4f}", flush=True)
    predictions.sort(key=lambda p: (p["example_id"], p["arm"]))
    write(output / "predictions.json", predictions)
    write(output / "prediction_freeze.json", {"created_at": now(),
          "predictions_sha256": sha(output / "predictions.json"),
          "calls": {p.name: sha(p) for p in sorted((output / "calls").iterdir())},
          "note": "All attempted test forecasts persisted before any scoring. Errors retained; no best-of selection."})
    return {"forecasts": len(predictions), "successful": sum(p["predicted_accuracy"] is not None for p in predictions),
            "cost_usd_reported": sum(p["cost_usd_reported"] for p in predictions)}


def score(bundle=DEFAULT_BUNDLE, output=DEFAULT_OUTPUT):
    bundle, output = Path(bundle), Path(output)
    policy, prompts = validate_prompt_bundle(bundle)
    freeze = read(output / "prediction_freeze.json")
    if sha(output / "predictions.json") != freeze["predictions_sha256"]:
        raise ValueError("Changed forecasts after freeze")
    verify_files(output / "calls", freeze["calls"])
    forecast_rows = read(output / "predictions.json")
    forecasts = {(p["example_id"], p["arm"]): p for p in forecast_rows}
    if len(forecasts) != len(forecast_rows) or set(forecasts) != {(p["example_id"], p["arm"]) for p in prompts}:
        raise ValueError("Forecast identity mismatch")
    for packet in prompts:
        pred = forecasts[(packet["example_id"], packet["arm"])]
        if pred["prompt_sha256"] != packet["prompt_sha256"]:
            raise ValueError("Forecast prompt identity mismatch")
        calls = [read(output / "calls" / (prediction_key(packet) + f"__attempt{i}.json"))
                 for i in range(1, pred["attempts"] + 1)]
        if any(c["example_id"] != packet["example_id"] or c["arm"] != packet["arm"]
               or c["prompt_sha256"] != packet["prompt_sha256"] for c in calls):
            raise ValueError("Call identity mismatch")
        successes = [c for c in calls if c.get("prediction") is not None]
        if len(successes) > 1 or (successes and successes[0] is not calls[-1]):
            raise ValueError("Not the first successful forecast")
        expected = None if not successes else successes[0]["prediction"]["predicted_accuracy"]
        if pred["predicted_accuracy"] != expected or pred["error"] != calls[-1]["error"]:
            raise ValueError("Forecast differs from saved inference response")
        if pred["predicted_accuracy"] is not None:
            result_events = [e for e in calls[-1]["raw_events"] if e.get("type") == "result"]
            parsed = parse_prediction(result_events[-1]["result"])
            if parsed != calls[-1]["prediction"]:
                raise ValueError("Forecast differs from raw model response")
            if pred["error"] or pred["predicted_delta"] != pred["predicted_accuracy"] - packet["parent_reference"]:
                raise ValueError("Invalid delta forecast")
    training, regressors = Path(policy["training"]), Path(policy["regressors"])
    if sha(regressors / "manifest.json") != policy["regressor_manifest_sha256"]:
        raise ValueError("Changed regression result manifest")
    verify_files(regressors, read(regressors / "manifest.json"))
    rows, labels = read(training / "inputs.json"), read(training / "labels.json")
    frozen_regressors = read(regressors / "predictions.json")
    report = {"scored_at": now(), "prediction_frozen_at": freeze["created_at"],
              "primary_llm_arm": policy["primary_llm_arm"], "benchmarks": {},
              "cost_usd_reported": sum(p["cost_usd_reported"] for p in forecast_rows),
              "note": "Both arms and all regressors use common successful IDs within each benchmark. Errors/coverage reported. Test explored previously."}
    for benchmark in TASKS:
        intended = [r for r in rows if r["split"] == "test" and r["benchmark"] == benchmark]
        chosen = [r for r in intended if all(forecasts[(r["example_id"], a)]["predicted_accuracy"] is not None for a in ARMS)]
        ids = [r["example_id"] for r in chosen]
        y = np.asarray([labels[k]["accuracy"] for k in ids])
        reference = np.asarray([r["parent_reference"] for r in chosen])
        ps = {a: np.asarray([forecasts[(k, a)]["predicted_accuracy"] for k in ids]) for a in ARMS}
        legacy = frozen_regressors["combined__" + benchmark]
        for arm in next(iter(legacy.values())):
            ps["regressor::" + arm] = np.asarray([legacy[k][arm] for k in ids])
        metrics = {}
        for arm, values in ps.items():
            metrics[arm] = {}
            for subset in ("all", "published_base", "measured_parent"):
                mask = np.asarray([subset == "all" or r["reference_kind"] == subset for r in chosen], dtype=bool)
                subrows = [r for r, keep in zip(chosen, mask) if keep]
                metrics[arm][subset] = {"accuracy": evaluate(subrows, y[mask], values[mask]),
                                       "delta": evaluate(subrows, (y-reference)[mask], (values-reference)[mask])}
        report["benchmarks"][benchmark] = {
            "intended_test_n": len(intended), "common_success_n": len(chosen), "common_ids": ids,
            "arm_success_counts": {a: sum(forecasts[(r["example_id"], a)]["predicted_accuracy"] is not None for r in intended) for a in ARMS},
            "metrics": metrics,
            "selected_regressor_vs_llm": {a: bootstrap_gain(chosen, y, ps["regressor::train_selected"], ps[a]) for a in ARMS} if chosen else {},
            "fixed_hgb_history_vs_llm": {a: bootstrap_gain(chosen, y, ps["regressor::hgb_history"], ps[a]) for a in ARMS} if chosen else {},
        }
    write(output / "report.json", report)
    write(output / "manifest.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--regressors", type=Path, default=DEFAULT_REGRESSORS)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.training, args.regressors, args.bundle)
    elif args.action == "run":
        result = run(args.bundle, args.output)
    else:
        report = score(args.bundle, args.output)
        result = {b: {"n": v["common_success_n"], "mae": {a: v["metrics"][a]["all"]["accuracy"]["mae"] for a in (*ARMS, "regressor::train_selected", "regressor::hgb_current", "regressor::hgb_history")}} for b, v in report["benchmarks"].items()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
