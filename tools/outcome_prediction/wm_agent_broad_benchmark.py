"""Frozen dataset-aware, broad-evidence agent with an accuracy/delta choice."""

from __future__ import annotations

import argparse
import copy
import math
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tools.outcome_prediction import wm_agent_local_benchmark as old
from tools.outcome_prediction import wm_agent_local_recovery as recovery
from tools.outcome_prediction.wm_clean_refresh import read, sha, write

ROOT = Path("data/analysis/wm_agent_local")
BUNDLE = ROOT / "v2_bundle"
OUTPUT = ROOT / "v2_results"
SOURCE = Path("data/analysis/wm_delta_reference/01406da734fb_v1")
SELECT_SYSTEM = """You design a lightweight world-model predictor for the outcome
of a proposed post-training experiment. Choose evidence weights, a target, model,
and a few features. A numerical tool fits your choice; its forecast is final.
You have no tools and must not output a numeric prediction for the query.

Crucial distinction: the EVALUATION BENCHMARK is NOT the TRAINING DATASET.
Training on GSM8K, MetaMath, OpenR1, NuminaMath, other named corpora, or a mixture
can produce different outcomes even with the same optimizer settings. Consider
domain/difficulty match, synthetic/derived data, possible data shifts, loss masks,
training dose and context length. Dataset evidence is a prospective declaration
or screened code reference, NOT proof that a dataset or mixture was actually used.
Dataset family names do not resolve exact versions or revisions.
Unknown identity or quantity means unknown, not zero, not all data, not the
evaluation dataset. Multiple references do not establish a realized mixture.
Do not invent quality, contamination, sample counts, mixture ratios or outcomes.

All clean labeled TRAIN experiments on this benchmark/base family are supplied.
Use ALL of them. Do not select a tiny convenient subset or cherry-pick successes.
Give relevant data more weight, considering dataset identity, parent strength,
base-start versus continuation, training method and history, but retain failures,
zero scores and negative deltas. A reasonable experiment can fail. Broad evidence
reduces instability; unrelated datasets can still cause bias and should be
distinguished through relevant feature columns and restrained reweighting.

Every example defaults to relevance weight 1. You may override any alias with a
weight in [0.5,2]. No exclusion/zero weights. The fitter adjusts for repeated
examples within each training session, then mixes the normalized agent weights
50/50 with uniform session-balanced weights. This deliberately preserves broad
support rather than pretending a few near neighbors are reliable evidence.
Opaque aliases/session aliases are handles, never predictive identities.

Choose target="accuracy" or target="delta" using only TRAIN evidence and the
query's known parent/recipe. For accuracy, fit the final official score directly.
For delta, fit official score minus known parent, then add query parent back.
Final accuracy is clipped to [0,1] in both cases. With its unpenalized intercept,
Ridge delta shrinks toward query parent plus weighted TRAIN mean delta, NOT
necessarily an unchanged parent. Absolute Ridge shrinks toward weighted TRAIN
mean accuracy. Learned parent-dependent coefficients can modify these tendencies;
the choices may suit different regimes.
When all parents are the same fixed base, the two targets can be equivalent for
an intercept model. Do not force delta, accuracy, or positive improvements.

Choose one model: weighted_median, ridge_10, ridge_100, shallow_tree.
weighted_median is a robust constant of the chosen target, but a single constant
can be biased when parent strengths or training datasets differ. Prefer a small
regularized regression when conditional structure is credible; do not default to
median merely because the pool is small. Ridge has train-only weighted scaling,
an intercept and L2 alpha10 or100. Tree has depth2, minimum leaf5. For learned
models choose 1-12 exact feature_order names INCLUDING parent.accuracy; consider
dataset indicators, missingness masks, and a few training/history settings.
Do not select many redundant columns. Median may use an empty feature list.

The feature_order decodes complete numeric vectors. A setting with observed=0
is unknown even if its encoded value is0. Parent accuracy is a real supplied
score or a published fixed-base reference, distinguished by reference flags.
Zero scores are valid. History contains configurations, never ancestor outcomes.
No fitting results, validation errors or query outcomes will be returned to you.

Return one JSON object with exactly these fields:
{"weight_overrides": {"T001": 1.5}, "target": "accuracy",
 "model": "ridge_10", "feature_keys": ["parent.accuracy"],
 "rationale": "Explain dataset compatibility, broad support, target and model choice in <=160 words."}
The example is a schema illustration, not a recommendation. An empty override
object means all weights1. Use real aliases/feature names supplied in context.
No selected subset, numeric query forecast, markdown, or extra prose.
"""
PREDICT_SYSTEM = """Predict the final official accuracy of a proposed post-training
experiment using supplied clean labeled training experiments and query features.
You have no tools. Return one JSON object with predicted_accuracy (number in
[0,1]) and rationale (brief explanation). Never return percent units like75.

The evaluation task is distinct from each recipe's training datasets. Use the
explicit structured dataset evidence, settings and prior recipe history; unknown
source/quantity is not absence or full-data training. Dataset references are not
proof of actual exposure or mixture proportions. Sensible intent does not assure
success. Retain evidence of zeros/failures/degradation and consider parent strength,
base-start/continuation and dataset/method shifts. An observed=0 setting is unknown.

You receive ALL clean same-benchmark training examples, not a tiny selected bank.
The trailing effective_weights array aligns with context.training_pool and can
guide relevance/calibration; it is session adjusted and blended with broad uniform
support. You have the same factual inputs and weights as a separate numerical
predictor, but you do not see its model, target choice, feature selection, rationale,
coefficients or forecast. Make your own prediction. Opaque aliases are not features.
The query's parent accuracy is known; its final accuracy and delta are not supplied.
"""
RESPONSE_KEYS = frozenset({"weight_overrides", "target", "model", "feature_keys", "rationale"})


def source_hashes():
    names = ("wm_agent_broad_benchmark.py", "wm_agent_broad_fit.py", "wm_agent_recipe_data.py",
             "wm_agent_local_benchmark.py", "wm_agent_local_fit.py", "wm_agent_local_recovery.py",
             "wm_local_agent_io.py", "wm_fresh_llm_baseline.py", "wm_clean_refresh.py",
             "wm_one_step_train.py", "wm_one_step_features.py", "wm_code_benchmark.py", "wm_small_benchmark.py")
    return {str((Path(__file__).parent / name).resolve()): sha(Path(__file__).parent / name) for name in names}


def verify_recipe_source():
    manifest = read(SOURCE / "manifest.json")
    old.previous.verify_files(SOURCE, manifest["files"])
    for path, expected in manifest["sources"].items():
        if sha(path) != expected:
            raise ValueError("Changed screened recipe source dependency")


def vector(row, order):
    from tools.outcome_prediction.wm_agent_broad_fit import FEATURE_KEYS, validate_row

    if list(order) != list(FEATURE_KEYS):
        raise ValueError("Feature order must match the exact frozen augmented schema")
    validate_row(row)
    values = row["views"]["history"]
    if set(values) != set(order):
        raise ValueError("Augmented schema mismatch")
    result = [values[key] for key in order]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in result):
        raise ValueError("Invalid augmented feature value")
    return result


def compact_context(context):
    """Keep dataset evidence and uncertainty; centralize fixed warning boilerplate."""
    step_keys = ("status", "families", "evidence", "multiple_family_references",
                 "mixture_declaration", "synthetic_declaration", "derived_or_replay_declaration",
                 "unmapped_declared_source_present")
    history_keys = ("history_family_union", "history_recipe_count", "history_complete_to_base",
                    "history_has_unresolved_dataset_identity")
    return {**{scope: {key: context[scope][key] for key in step_keys}
               for scope in ("current", "immediate_parent_recipe")},
            **{key: context[key] for key in history_keys}}


def prompt_payload(query, pool, labels, aliases, evidence, order):
    if any(r["split"] != "train" or r["benchmark"] != query["benchmark"] for r in pool):
        raise ValueError("Candidate pool must be same-benchmark TRAIN only")
    if query["cell_id"] in {r["cell_id"] for r in pool} or aliases != old.aliases_for(pool):
        raise ValueError("Unsafe query/session/alias mapping")
    old.validate_rows(pool, labels)
    old.validate_rows([query])
    by_id = {r["example_id"]: r for r in pool}
    contexts, context_keys = {}, {}
    for info in aliases.values():
        compact = compact_context(evidence[info["example_id"]]["context"])
        signature = old.canonical(compact)
        if signature not in context_keys:
            code = f"D{len(context_keys) + 1:03d}"
            context_keys[signature] = code
            contexts[code] = compact
    # Context is identical across queries within a benchmark and sorts before
    # query. Query-specific weights are appended separately, not inside rows.
    return {
        "context": {
            "task": old.previous.TASKS[query["benchmark"]], "feature_order": order,
            "dataset_contexts": contexts,
            "training_pool": [{
                "alias": alias, "session_alias": info["session_alias"],
                "features": vector(by_id[info["example_id"]], order),
                "dataset_context_key": context_keys[old.canonical(compact_context(evidence[info["example_id"]]["context"]))],
                "official_accuracy": labels[info["example_id"]]["accuracy"],
                "official_delta": labels[info["example_id"]]["delta_accuracy"],
            } for alias, info in aliases.items()],
        },
        "query": {"features": vector(query, order), "dataset_context": compact_context(evidence[query["example_id"]]["context"])},
    }


def resolve_spec(response, aliases, pool, query):
    from tools.outcome_prediction.wm_agent_broad_fit import validate_spec

    if not isinstance(response, dict) or not RESPONSE_KEYS.issubset(response):
        raise ValueError("Required broad-model schema fields missing")
    overrides = response["weight_overrides"]
    if not isinstance(overrides, dict) or set(overrides) - set(aliases):
        raise ValueError("Unknown alias or non-object weight overrides")
    alias_of = {info["example_id"]: alias for alias, info in aliases.items()}
    spec = {"weights": [overrides.get(alias_of[row["example_id"]], 1.0) for row in pool],
            **{key: response[key] for key in RESPONSE_KEYS - {"weight_overrides"}}}
    return validate_spec(spec, pool, query), {key: value for key, value in response.items() if key not in RESPONSE_KEYS}


def direct_prompt(payload, pool, aliases, weights):
    if aliases != old.aliases_for(pool) or [r["alias"] for r in payload["context"]["training_pool"]] != list(aliases):
        raise ValueError("Direct-LLM alias/row/weight alignment mismatch")
    if len(weights) != len(pool) or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Invalid full-pool direct-LLM weights")
    by_id = {row["example_id"]: weight for row, weight in zip(pool, weights)}
    return old.canonical({**payload, "weighting": {"effective_weights": [by_id[info["example_id"]] for info in aliases.values()]}})


def prepare(bundle=BUNDLE):
    from tools.outcome_prediction.wm_agent_broad_fit import FEATURE_KEYS
    from tools.outcome_prediction.wm_agent_recipe_data import extract

    bundle = Path(bundle)
    if bundle.exists():
        raise FileExistsError("Use a new immutable bundle")
    _, old_train, labels, old_queries, banks, _ = recovery.load_blinded_bundle(old.BUNDLE)
    verify_recipe_source()
    examples = read(SOURCE / "inputs.json")
    ids = {r["example_id"] for r in old_train + old_queries}
    if len(examples) != len(ids) or {e["example_id"] for e in examples} != ids:
        raise ValueError("Screened recipe/clean row membership mismatch")
    evidence = {e["example_id"]: extract(e) for e in examples}
    train, queries = copy.deepcopy(old_train), copy.deepcopy(old_queries)
    for row in train + queries:
        row["views"]["history"].update(evidence[row["example_id"]]["features"])
        vector(row, FEATURE_KEYS)
    if len(train) != 123 or len(queries) != 62:
        raise ValueError("Expected unchanged clean123/62 split")
    linked_files = [old.BUNDLE / "manifest.json", SOURCE / "manifest.json", SOURCE / "inputs.json",
                    ROOT / "v1_recovery" / "prediction_freeze.json", old.PREVIOUS / "manifest.json", old.PREVIOUS / "predictions.json",
                    old.TRAINING / "manifest.json", old.TRAINING / "labels.json", old.REGRESSORS / "manifest.json"]
    policy = {
        "schema": "dataset-aware-broad-agent-v2", "created_at": old.now(),
        "sources": source_hashes(), "linked_files": {str(p.resolve()): sha(p) for p in linked_files},
        "libraries": {name: version(name) for name in ("numpy", "scikit-learn")},
        "selector_system": SELECT_SYSTEM, "predictor_system": PREDICT_SYSTEM,
        "feature_order": list(FEATURE_KEYS), "model_requested": "claude-opus-5", "effort": "high",
        "timeout_seconds": 240, "max_cli_network_retries": 1, "workers": 4, "max_attempts": 2,
        "selector_max_output_tokens": 4096, "predictor_max_output_tokens": 2048,
        "selector_budget_usd": 1.25, "predictor_budget_usd": 1.0, "reported_cost_stop_usd": 100.0,
        "primary": "agent_target", "controls": ["fixed_accuracy", "fixed_delta", "uniform_weights", "broad_evidence_llm"],
        "target_choice": "One agent choice based only on TRAIN outcomes and query input; no fit, CV or held-out-error feedback. Same model/columns/weights refitted on each target for controls.",
        "broad_evidence": "All51GSM or72AIME cleanTRAINrows retained; relevance[0.5,2], per-session adjusted and separately normalized, then50/50 blended with normalized uniform session-balanced weights.",
        "prompt": "Same augmented vectors and structured prospective dataset context for selector/directLLM. Direct sees actual weights, not selector target/model/features/rationale/forecast. Shared context first, query/weights last.",
        "parsing": "Required fields projected from first semantically valid response; ignored extra metadata saved. At most2 attempts, same prompt, no model-quality feedback. No postscore recovery.",
        "metric": "Equal-session MAE per benchmark on identical common successfulIDs; accuracy/deltaR2, reference-kind subsets, fixed heldout session bootstrap and all failure coverage.",
        "limitations": ["Previously explored developmental split; not independent confirmation.", "Dataset features, prompt, broad weights and targetchoice change together; comparison toV1 is whole-pipeline, not a dataset-feature causal ablation.", "Matched directLLM sees all augmented columns; learned model uses<=12 agent-chosen columns.", "Conditional TRAIN LOSO is not unbiased evaluation of label-informed agent policy.", "Costs are CLI equivalents, not subscription invoices. Selector remains LLM-dependent."],
    }
    packets = []
    for query in queries:
        pool = [r for r in train if r["benchmark"] == query["benchmark"]]
        payload = prompt_payload(query, pool, {r["example_id"]: labels[r["example_id"]] for r in pool}, banks[query["benchmark"]], evidence, FEATURE_KEYS)
        prompt = old.canonical(payload)
        packets.append({"example_id": query["example_id"], "prompt": prompt, "prompt_sha256": old.digest(prompt)})
    bundle.mkdir(parents=True, mode=0o700)
    for name, value in (("train_inputs.json", train), ("train_labels.json", labels), ("queries.json", queries),
                        ("aliases.json", banks), ("dataset_evidence.json", evidence), ("selector_prompts.json", packets), ("policy.json", policy)):
        write(bundle / name, value)
    write(bundle / "manifest.json", {p.name: sha(p) for p in bundle.iterdir() if p.is_file()})
    validate_bundle(bundle)
    return {"train": len(train), "test": len(queries), "feature_count": len(FEATURE_KEYS),
            "prompt_chars": [min(len(p["prompt"]) for p in packets), max(len(p["prompt"]) for p in packets)]}


def validate_bundle(bundle):
    from tools.outcome_prediction.wm_agent_broad_fit import FEATURE_KEYS
    from tools.outcome_prediction.wm_agent_recipe_data import extract

    bundle = Path(bundle)
    manifest = read(bundle / "manifest.json")
    if set(manifest) != {"train_inputs.json", "train_labels.json", "queries.json", "aliases.json", "dataset_evidence.json", "selector_prompts.json", "policy.json"}:
        raise ValueError("Unexpected v2 bundle files")
    old.previous.verify_files(bundle, manifest)
    policy = read(bundle / "policy.json")
    if policy["sources"] != source_hashes() or policy["feature_order"] != list(FEATURE_KEYS):
        raise ValueError("Changed frozen source/schema")
    for path, expected in policy["linked_files"].items():
        if sha(path) != expected:
            raise ValueError("Changed upstream input/result manifest")
    if policy["libraries"] != {name: version(name) for name in policy["libraries"]}:
        raise ValueError("Changed numerical library versions")
    if policy["selector_system"] != SELECT_SYSTEM or policy["predictor_system"] != PREDICT_SYSTEM:
        raise ValueError("Changed system prompts")
    verify_recipe_source()
    _, old_train, labels, old_queries, banks, _ = recovery.load_blinded_bundle(old.BUNDLE)
    expected = {e["example_id"]: extract(e) for e in read(SOURCE / "inputs.json")}
    evidence = read(bundle / "dataset_evidence.json")
    if evidence != expected:
        raise ValueError("Dataset evidence not reproducible from screened inputs")
    train, queries = copy.deepcopy(old_train), copy.deepcopy(old_queries)
    for row in train + queries:
        row["views"]["history"].update(expected[row["example_id"]]["features"])
    if (train != read(bundle / "train_inputs.json") or queries != read(bundle / "queries.json")
            or labels != read(bundle / "train_labels.json") or banks != read(bundle / "aliases.json")):
        raise ValueError("Changed source-derived clean examples/labels/aliases")
    packets = read(bundle / "selector_prompts.json")
    if len(packets) != len(queries):
        raise ValueError("Missing selector prompts")
    for query, packet in zip(queries, packets):
        pool = [r for r in train if r["benchmark"] == query["benchmark"]]
        prompt = old.canonical(prompt_payload(query, pool, {r["example_id"]: labels[r["example_id"]] for r in pool}, banks[query["benchmark"]], evidence, FEATURE_KEYS))
        if packet != {"example_id": query["example_id"], "prompt": prompt, "prompt_sha256": old.digest(prompt)}:
            raise ValueError("Selector prompt alignment/reconstruction mismatch")
    return policy, train, labels, queries, banks, evidence, packets


def run(bundle=BUNDLE, output=OUTPUT):
    from tools.outcome_prediction.wm_agent_broad_fit import fit_predict

    bundle, output = Path(bundle), Path(output)
    policy, train, labels, queries, banks, evidence, packets = validate_bundle(bundle)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name in ("calls", "direct_prompts", "query_results"):
        (output / name).mkdir(mode=0o700)
    write(output / "run_policy.json", {"started_at": old.now(), "bundle_manifest_sha256": sha(bundle / "manifest.json"),
          "claude_version": subprocess.run(["claude", "--version"], capture_output=True, text=True, check=True).stdout.strip()})
    budget = recovery.CallBudget(policy["reported_cost_stop_usd"], max(policy["selector_budget_usd"], policy["predictor_budget_usd"]))
    started = time.perf_counter()

    def invoke(prompt, stage, system=None):
        call_policy = {**policy, "system": system or policy[stage + "_system"],
                       "max_output_tokens": policy[stage + "_max_output_tokens"], "per_call_budget_usd": policy[stage + "_budget_usd"]}
        return recovery._invoke(prompt, call_policy, budget)

    probe = invoke('Return exactly {"ok":true}.', "selector", "Connectivity check. Return the requested JSON object. No tools.")
    write(output / "probe.json", probe)
    if probe.get("error") or probe.get("response") != {"ok": True}:
        raise RuntimeError("Connectivity/isolation probe failed")
    packet_of = {p["example_id"]: p for p in packets}

    def work(query):
        key, identity = old.key_for(query), query["example_id"]
        pool = [r for r in train if r["benchmark"] == query["benchmark"]]
        pool_labels = {r["example_id"]: labels[r["example_id"]] for r in pool}
        packet = packet_of[identity]
        result = {"example_id": identity, "benchmark": query["benchmark"], "selector_prompt_sha256": packet["prompt_sha256"],
                  "spec": None, "fits": None, "llm_prediction": None, "error": None}
        for attempt in range(1, policy["max_attempts"] + 1):
            record = invoke(packet["prompt"], "selector")
            if record.get("response") is not None and not record.get("error"):
                try:
                    result["spec"], record["ignored_fields"] = resolve_spec(record["response"], banks[query["benchmark"]], pool, query)
                except (ValueError, TypeError, KeyError) as exc:
                    record["error"] = "invalid_spec: " + str(exc)
            record.update(example_id=identity, stage="selector", attempt=attempt, prompt_sha256=packet["prompt_sha256"])
            write(output / "calls" / (key + f"__selector__attempt{attempt}.json"), record)
            result["selector_attempts"] = attempt
            if result["spec"] is not None or not record.get("attempted") or budget.stopped:
                break
        if result["spec"] is None:
            result["error"] = record["error"] or "no_valid_spec"
            write(output / "query_results" / (key + ".json"), result)
            return result
        spec = result["spec"]
        try:
            local = fit_predict(pool, pool_labels, query, spec)
            fits = {"agent_target": local}
            for target in ("accuracy", "delta"):
                fits["fixed_" + target] = local if spec["target"] == target else fit_predict(pool, pool_labels, query, {**spec, "target": target})
            fits["uniform_weights"] = fit_predict(pool, pool_labels, query, {**spec, "weights": [1.0] * len(pool)})
            result["fits"] = fits
            payload = prompt_payload(query, pool, pool_labels, banks[query["benchmark"]], evidence, policy["feature_order"])
            weights = local["fit_metadata"]["normalized_sample_weights"]
            prompt = direct_prompt(payload, pool, banks[query["benchmark"]], weights)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            result["error"] = "fit_error: " + str(exc)
            write(output / "query_results" / (key + ".json"), result)
            return result
        prompt_hash = old.digest(prompt)
        write(output / "direct_prompts" / (key + ".json"), {"example_id": identity, "prompt": prompt, "prompt_sha256": prompt_hash})
        for attempt in range(1, policy["max_attempts"] + 1):
            record = invoke(prompt, "predictor")
            if record.get("response") is not None and not record.get("error"):
                try:
                    result["llm_prediction"] = old.previous.parse_prediction(old.canonical(record["response"]))
                except (ValueError, TypeError, KeyError) as exc:
                    record["error"] = "invalid_forecast: " + str(exc)
            record.update(example_id=identity, stage="predictor", attempt=attempt, prompt_sha256=prompt_hash)
            write(output / "calls" / (key + f"__predictor__attempt{attempt}.json"), record)
            result["predictor_attempts"] = attempt
            if result["llm_prediction"] is not None or not record.get("attempted") or budget.stopped:
                break
        result["error"] = None if result["llm_prediction"] is not None else record["error"]
        write(output / "query_results" / (key + ".json"), result)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=policy["workers"]) as executor:
        pending = [executor.submit(work, q) for q in queries]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            spec = result["spec"] or {}
            print(f"{len(results)}/{len(queries)} {result['example_id']} target={spec.get('target')} model={spec.get('model')} status={result['error'] or 'ok'} cost={budget.reported:.2f}", flush=True)
    results.sort(key=lambda r: r["example_id"])
    write(output / "predictions.json", results)
    write(output / "prediction_freeze.json", {"created_at": old.now(), "elapsed_seconds": time.perf_counter() - started,
          "reported_cost_usd": budget.reported, "charged_cost_usd": budget.charged, "systemic_stop": budget.stopped,
          "files": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
          "note": "Every choice/fit/control/direct forecast frozen before v2 target scoring; no test labels in inference or fitting."})
    return {"queries": len(results), "local_success": sum(r["fits"] is not None for r in results),
            "llm_success": sum(r["llm_prediction"] is not None for r in results), "reported_cost_usd": budget.reported}


def score(bundle=BUNDLE, output=OUTPUT):
    from tools.outcome_prediction.wm_agent_broad_fit import fit_predict

    bundle, output = Path(bundle), Path(output)
    if (output / "report.json").exists():
        raise FileExistsError("Preserve immutable scored results")
    freeze = read(output / "prediction_freeze.json")
    old.previous.verify_files(output, freeze["files"])
    policy, train, labels, queries, banks, evidence, packets = validate_bundle(bundle)
    if read(output / "run_policy.json")["bundle_manifest_sha256"] != sha(bundle / "manifest.json"):
        raise ValueError("Forecast run was linked to a different bundle")
    predictions = read(output / "predictions.json")
    values = {p["example_id"]: p for p in predictions}
    if len(values) != len(predictions) or set(values) != {q["example_id"] for q in queries}:
        raise ValueError("Missing or duplicate prediction identities")
    packet_of = {p["example_id"]: p for p in packets}
    for query in queries:
        identity, key = query["example_id"], old.key_for(query)
        result = values[identity]
        if result != read(output / "query_results" / (key + ".json")) or result["selector_prompt_sha256"] != packet_of[identity]["prompt_sha256"]:
            raise ValueError("Prediction/query/selector mismatch")
        if result["spec"] is None:
            continue
        pool = [r for r in train if r["benchmark"] == query["benchmark"]]
        pool_labels = {r["example_id"]: labels[r["example_id"]] for r in pool}
        calls = [read(output / "calls" / (key + f"__selector__attempt{i}.json")) for i in range(1, result["selector_attempts"] + 1)]
        valid = [c for c in calls if not c.get("error")]
        if len(valid) != 1 or valid[0] is not calls[-1]:
            raise ValueError("Selection was not first valid response")
        for i, call in enumerate(calls, 1):
            if call["example_id"] != identity or call["stage"] != "selector" or call["attempt"] != i or call["prompt_sha256"] != packet_of[identity]["prompt_sha256"]:
                raise ValueError("Selector call alignment mismatch")
        old.validate_success_call(valid[0], policy)
        spec, ignored = resolve_spec(valid[0]["response"], banks[query["benchmark"]], pool, query)
        if spec != result["spec"] or ignored != valid[0].get("ignored_fields"):
            raise ValueError("Changed selection projection")
        if result["fits"] is None:
            continue
        specs = {"agent_target": spec, "fixed_accuracy": {**spec, "target": "accuracy"},
                 "fixed_delta": {**spec, "target": "delta"}, "uniform_weights": {**spec, "weights": [1.0] * len(pool)}}
        for arm, selected_spec in specs.items():
            if fit_predict(pool, pool_labels, query, selected_spec) != result["fits"][arm]:
                raise ValueError("Saved training-only fit/control fails exact replay")
        payload = prompt_payload(query, pool, pool_labels, banks[query["benchmark"]], evidence, policy["feature_order"])
        prompt = direct_prompt(payload, pool, banks[query["benchmark"]], result["fits"]["agent_target"]["fit_metadata"]["normalized_sample_weights"])
        saved = read(output / "direct_prompts" / (key + ".json"))
        if saved != {"example_id": identity, "prompt": prompt, "prompt_sha256": old.digest(prompt)}:
            raise ValueError("Direct-LLM factual context/weight mismatch")
        if result["llm_prediction"] is not None:
            calls = [read(output / "calls" / (key + f"__predictor__attempt{i}.json")) for i in range(1, result["predictor_attempts"] + 1)]
            valid = [c for c in calls if not c.get("error")]
            if len(valid) != 1 or valid[0] is not calls[-1]:
                raise ValueError("Direct forecast not first valid response")
            for i, call in enumerate(calls, 1):
                if call["example_id"] != identity or call["stage"] != "predictor" or call["attempt"] != i or call["prompt_sha256"] != old.digest(prompt):
                    raise ValueError("Direct call/query/weight mismatch")
            old.validate_success_call(valid[0], policy)
            if old.previous.parse_prediction(old.canonical(valid[0]["response"])) != result["llm_prediction"]:
                raise ValueError("Changed direct forecast")
    # Only scoring, after prediction freeze and deterministic training-only replay.
    targets = read(old.TRAINING / "labels.json")
    fixed = {(r["example_id"], r["arm"]): r for r in read(old.PREVIOUS / "predictions.json")}
    v1_dir = ROOT / "v1_recovery"
    old.previous.verify_files(v1_dir, read(v1_dir / "prediction_freeze.json")["files"])
    v1 = {r["example_id"]: r for r in read(v1_dir / "predictions.json")}
    old.previous.verify_files(old.REGRESSORS, read(old.REGRESSORS / "manifest.json"))
    reg = read(old.REGRESSORS / "predictions.json")
    report = {"scored_at": old.now(), "prediction_frozen_at": freeze["created_at"], "reported_cost_usd": freeze["reported_cost_usd"],
              "benchmarks": {}, "limitations": policy["limitations"]}
    for benchmark in old.previous.TASKS:
        intended = [q for q in queries if q["benchmark"] == benchmark]
        common = [q for q in intended if values[q["example_id"]]["fits"] is not None and values[q["example_id"]]["llm_prediction"] is not None]
        ids = [q["example_id"] for q in common]
        y = np.asarray([targets[k]["accuracy"] for k in ids])
        parent = np.asarray([q["parent_reference"] for q in common])
        forecasts = {**{arm: np.asarray([values[k]["fits"][arm]["predicted_accuracy"] for k in ids]) for arm in ("agent_target", "fixed_accuracy", "fixed_delta", "uniform_weights")},
                     "broad_evidence_llm": np.asarray([values[k]["llm_prediction"]["predicted_accuracy"] for k in ids]),
                     "v1_agent_local": np.asarray([v1[k]["local_fit"]["predicted_accuracy"] for k in ids]),
                     "v1_selected_llm": np.asarray([v1[k]["llm_prediction"]["predicted_accuracy"] for k in ids]),
                     "fixed_few_shot": np.asarray([fixed[k, "few_shot"]["predicted_accuracy"] for k in ids]),
                     **{arm: np.asarray([reg["combined__" + benchmark][k][arm] for k in ids]) for arm in ("train_selected", "hgb_current", "hgb_history")}}
        metrics = {}
        for arm, pred in forecasts.items():
            metrics[arm] = {}
            for subset in ("all", "published_base", "measured_parent"):
                mask = np.asarray([subset == "all" or q["reference_kind"] == subset for q in common], dtype=bool)
                rows = [q for q, keep in zip(common, mask) if keep]
                metrics[arm][subset] = {"accuracy": old.evaluate(rows, y[mask], pred[mask]), "delta": old.evaluate(rows, (y - parent)[mask], (pred - parent)[mask])}
        comparison_arms = ("fixed_few_shot", "broad_evidence_llm", "v1_agent_local", "fixed_accuracy", "fixed_delta", "uniform_weights")
        report["benchmarks"][benchmark] = {"intended": len(intended), "common_success": len(common), "common_ids": ids,
            "coverage": {"local_fit": sum(values[q["example_id"]]["fits"] is not None for q in intended), "direct_llm": sum(values[q["example_id"]]["llm_prediction"] is not None for q in intended)},
            "target_choices": dict(Counter(values[k]["spec"]["target"] for k in ids)), "model_choices": dict(Counter(values[k]["spec"]["model"] for k in ids)),
            "metrics": metrics, "primary_comparisons": {arm: old.compare_fixed_holdout(common, y, forecasts["agent_target"], forecasts[arm]) for arm in comparison_arms} if common else {}}
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
        result = prepare(args.bundle)
    elif args.action == "run":
        result = run(args.bundle, args.output)
    else:
        report = score(args.bundle, args.output)
        result = {b: {"n": v["common_success"], "mae_pp": {arm: round(m["all"]["accuracy"]["mae"] * 100, 3) if m["all"]["accuracy"]["mae"] is not None else None for arm, m in v["metrics"].items()}} for b, v in report["benchmarks"].items()}
    print(old.canonical(result))


if __name__ == "__main__":
    main()
