"""Frozen, outcome-blind matching pilot; never alter earlier experiment artifacts."""

from __future__ import annotations

import argparse
import math
import subprocess
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tools.outcome_prediction import wm_agent_broad_benchmark as prior
from tools.outcome_prediction import wm_agent_local_benchmark as old
from tools.outcome_prediction import wm_agent_local_recovery as recovery
from tools.outcome_prediction.wm_clean_refresh import read, sha, write

ROOT = Path("data/analysis/wm_matched_pilot")
BUNDLE = ROOT / "v1_bundle"
OUTPUT = ROOT / "v1_results"
VARIANTS = ("all_rows", "state_only", "state_tag", "state_treatment_soft", "state_treatment_soft_shrunk")
MATCH_SYSTEM = """Choose historical experiments comparable to a proposed post-training
intervention on a known checkpoint. You have no tools. This is MATCHING ONLY:
candidate parent scores are given, but candidate final outcomes/deltas are hidden.
Do not predict the query outcome, select a numerical model, or invent results.

Compare starting state AND treatment: parent strength, preceding training recipes
and datasets, current training-data source, sampled/verified/wrong-attempt/replay
evidence, dose, loss settings, and current versus earlier configurations.
The recorded SFT/RFT/distill family tags are not disjoint optimizer classes here:
RFT refers to sampled/verified-answer supervised training in this audited cohort;
some SFT-tag recipes also use this mechanism. Shared supervised objective alone
does not make two treatments comparable. Dataset references are declarations,
not proof of actual usage, exact versions, quantity, filtering, or mixture ratios.
Unknown means unknown. In particular correctness_filter=0 is no recognized
positive phrase, NOT proof of unfiltered data, even when its old observed mask is1.
The separate positive treatment-evidence fields are True or null, never false.

Only same-benchmark/base-family, same reference-kind examples with parent accuracy
within .20, state-distance<=.60, and no known objective conflict are supplied.
Unknown objective is allowed but is not verified compatibility. These conditions
do NOT guarantee comparability. Choose genuinely comparable rows, including enough
independent sessions when available. Do not force unrelated rows into a tiny pool.
You may select 0 to24 distinct aliases, each with relevance weight in[0.5,2].
The fitter divides weights by selected count in each session. Usable support is
at least6 rows,3 sessions, and session effective sample size>=2.5. This is an
engineering gate, not a guarantee of reliability. Insufficient support produces
an explicit unchanged-parent fallback. It is acceptable to select fewer or none
when adequate evidence is absent. Outcomes cannot be used to choose matches.

All feature names and structured current/parent dataset and treatment contexts
are supplied. Feature values with observed=0 are missing, except positive marker
flags use the caution above. Never use opaque example/session aliases as features.
Return one JSON object:
{"selected_aliases":["T001","T002"],"weights":[1.0,1.5],
 "rationale":"Explain state/treatment comparability and support in <=130 words."}
No extra prose. No target accuracy, target choice, model choice, or desired effect.
"""
DIRECT_SYSTEM = """Predict a proposed checkpoint's final official accuracy from
the supplied parent accuracy, prospective recipe/history inputs, and labeled
training demonstrations. You have no tools and do not fit or execute models.
Return JSON {"predicted_accuracy":0.5,"rationale":"brief explanation"}.
Accuracy must be in[0,1], not percent units. Final query outcome is hidden.

Reason about parent state AND proposed treatment; a strong existing checkpoint
need not benefit like a fixed-base start. Improvements, regressions and zeros are
all possible. Sampled-answer, verified-filter, wrong-attempt and replay declarations
can differ even when both methods are supervised training. Recorded SFT/RFT tags
are not exclusive optimizer classes in this cohort. Training dataset references
do not certify actual usage, mixture proportions, quantity, quality or success.
Positive treatment-evidence fields are True or null; null is unknown. In the old
numeric features, correctness_filter=0 means no recognized positive phrase, NOT
known absence of filtering, even if its observed mask is1. Other observed=0 numeric
settings are unknown, not real zeros. Parent accuracy is valid, including true0.

Demonstrations are either a fixed session-diverse bank or a matched neighborhood,
as declared. Effective weights are aligned with the rows and session-adjusted.
You do not receive the matching agent's rationale, fitted numerical prediction,
or the query label. Make your own forecast from these factual inputs. Aliases
are handles, not predictive identities.
"""


def source_hashes():
    result = prior.source_hashes()
    for name in ("wm_matched_pilot.py", "wm_matched_fit.py", "wm_matched_evidence.py",
                 "wm_code_data_features.py"):
        path = Path(__file__).parent / name
        result[str(path.resolve())] = sha(path)
    return result


def task_pool(train, query):
    return [r for r in train if r["benchmark"] == query["benchmark"]]


def pool_labels(pool, labels):
    return {r["example_id"]: labels[r["example_id"]] for r in pool}


def context(row, evidence, treatment):
    from tools.outcome_prediction.wm_matched_evidence import validate_context

    validate_context(treatment[row["example_id"]]["context"])
    return {"dataset": prior.compact_context(evidence[row["example_id"]]["context"]),
            "treatment": treatment[row["example_id"]]["context"]}


def packet(query, rows, evidence, treatment, order, *, labels=None, weights=None, kind="matching"):
    """One shared input whitelist; outcomes appear only in direct-LLM packets."""
    aliases = old.aliases_for(rows) if rows else {}
    by_id = {r["example_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate demonstration identity")
    if any(r["split"] != "train" or r["benchmark"] != query["benchmark"]
           or r["cell_id"] == query["cell_id"] for r in rows):
        raise ValueError("Unsafe demonstration/matching pool")
    if labels is not None and set(labels) != set(by_id):
        raise ValueError("Direct packet must contain only selected training labels")
    if weights is not None and len(weights) != len(rows):
        raise ValueError("Weight alignment mismatch")
    if labels is not None and weights is None:
        raise ValueError("Direct packet needs explicit aligned weights")
    if weights is not None and any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Weights must be finite and positive")
    if labels is not None:
        for row in rows:
            label = labels[row["example_id"]]
            if not isinstance(label, Mapping):
                raise TypeError("Missing official training label object")
            target = label.get("accuracy")
            delta = label.get("delta_accuracy")
            if (type(target) not in (int, float) or not math.isfinite(target) or not 0 <= target <= 1
                    or type(delta) not in (int, float) or not math.isfinite(delta)
                    or delta != target - row["parent_reference"]):
                raise ValueError("Invalid official target/parent/delta consistency")
    weight_of = dict(zip(by_id, weights)) if weights is not None else {}
    examples = []
    for alias, info in aliases.items():
        row = by_id[info["example_id"]]
        example = {"alias": alias, "session_alias": info["session_alias"],
                   "features": prior.vector(row, order), "context": context(row, evidence, treatment)}
        if labels is not None:
            example.update(official_accuracy=labels[row["example_id"]]["accuracy"],
                           official_delta=labels[row["example_id"]]["delta_accuracy"],
                           effective_weight=weight_of[row["example_id"]])
        examples.append(example)
    payload = {"task": old.previous.TASKS[query["benchmark"]], "feature_order": order,
               "example_kind": kind, "examples": examples,
               "query": {"features": prior.vector(query, order), "context": context(query, evidence, treatment)}}
    return old.canonical(payload), aliases


def resolve(response, aliases, pool, query, treatment):
    from tools.outcome_prediction.wm_matched_fit import selected_neighborhood

    fields = {"selected_aliases", "weights", "rationale"}
    if not isinstance(response, dict) or not fields.issubset(response):
        raise ValueError("Missing selection fields")
    names = response["selected_aliases"]
    if (not isinstance(names, list) or any(not isinstance(x, str) for x in names)
            or len(set(names)) != len(names) or set(names) - set(aliases)):
        raise ValueError("Invalid selection aliases")
    rationale = response["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 4000:
        raise ValueError("Invalid matching explanation")
    ids = [aliases[x]["example_id"] for x in names]
    neighborhood = selected_neighborhood(pool, query, ids, response["weights"], evidence=treatment)
    return {"selected_ids": ids, "weights": response["weights"], "rationale": rationale,
            "neighborhood": neighborhood}, {k: v for k, v in response.items() if k not in fields}


def uniform_session_weights(rows):
    counts = Counter(r["cell_id"] for r in rows)
    return [1.0 / counts[r["cell_id"]] for r in rows]


def prepare(bundle=BUNDLE):
    from tools.outcome_prediction.wm_agent_broad_fit import FEATURE_KEYS
    from tools.outcome_prediction.wm_matched_evidence import extract
    from tools.outcome_prediction.wm_matched_fit import candidate_rows, neighborhood

    bundle = Path(bundle)
    if bundle.exists():
        raise FileExistsError("Preserve previous frozen pilot")
    _, train, labels, queries, _, evidence, _ = prior.validate_bundle(prior.BUNDLE)
    source = read(prior.SOURCE / "inputs.json")
    if {x["example_id"] for x in source} != {x["example_id"] for x in train + queries}:
        raise ValueError("Treatment source membership mismatch")
    treatment = {x["example_id"]: extract(x) for x in source}
    pilot = [q for q in queries if q["reference_kind"] == "measured_parent"]
    banks = {b: old.previous.select_bank([r for r in train if r["benchmark"] == b], b)
             for b in old.previous.TASKS}
    packets, supports = [], []
    for query in queries:
        pool = task_pool(train, query)
        supports.append({"example_id": query["example_id"], "benchmark": query["benchmark"],
                         "neighborhoods": {v: neighborhood(pool, query, v, evidence=treatment) for v in VARIANTS}})
        if query not in pilot:
            continue
        candidates = candidate_rows(pool, query, evidence=treatment)
        prompt, aliases = packet(query, candidates, evidence, treatment, list(FEATURE_KEYS))
        bank = banks[query["benchmark"]]
        fixed, _ = packet(query, bank, evidence, treatment, list(FEATURE_KEYS), labels=pool_labels(bank, labels),
                          weights=uniform_session_weights(bank), kind="fixed_16_session_diverse")
        packets.append({"example_id": query["example_id"], "candidate_ids": [r["example_id"] for r in candidates],
                        "aliases": aliases, "selector": prompt, "selector_sha256": old.digest(prompt),
                        "fixed16": fixed, "fixed16_sha256": old.digest(fixed)})
    links = [prior.BUNDLE / "manifest.json", prior.SOURCE / "manifest.json", prior.SOURCE / "inputs.json",
             old.TRAINING / "labels.json", old.PREVIOUS / "predictions.json",
             old.REGRESSORS / "predictions.json", prior.ROOT / "v1_recovery/predictions.json",
             prior.ROOT / "v2_completion/predictions.json"]
    policy = {"created_at": old.now(), "schema": "matched-pilot-v1", "sources": source_hashes(),
              "linked_files": {str(p.resolve()): sha(p) for p in links}, "feature_order": list(FEATURE_KEYS),
              "variants": list(VARIANTS), "pilot_ids": [q["example_id"] for q in pilot],
              "scope": "All62 numerical forecasts + entire-procedure TRAIN LOSO; agent/fresh LLMs on all21 measured-parent TESTqueries, selected by inputs only.",
              "selection": "Outcome-blind agent matches0-24 hard-eligible training inputs; labels supplied only after matching. Fixed median/shrunk median, no model/target selection.",
              "validation": "Fixed-rule leave-one-TRAIN-session-out; no gates, models, matching or shrinkage chosen from its errors. Agent policy is not cross-validated in this limited pilot.",
              "support": "Engineering gate>=6rows,3sessions,sessionESS>=2.5; parent fallback on unsupported data, NULL on invocation failure. No filling across hard constraints.",
              "selector_system": MATCH_SYSTEM, "predictor_system": DIRECT_SYSTEM,
              "model_requested": "claude-opus-5", "effort": "high", "max_output_tokens": 3072,
              "per_call_budget_usd": 1.0, "reported_cost_stop_usd": 40.0,
              "max_cli_network_retries": 1, "timeout_seconds": 240, "workers": 4, "max_attempts": 2,
              "limitations": ["Previously explored developmental holdout, not new independent confirmation.",
                              "Support gates/calipers are fixed engineering choices, not statistical reliability certificates.",
                              "Recorded method-family matching is diagnostic; treatment evidence is positive declarations, not verified execution.",
                              "No local ridge or broad target/architecture search in this small-support pilot.",
                              "Fresh LLMs are21-query continuation pilot; older baselines use different feature/prompt contracts.",
                              "Sparse or distant neighborhoods fallback, rather than fabricate comparable evidence."]}
    bundle.mkdir(parents=True, mode=0o700)
    for name, value in (("train_inputs.json", train), ("train_labels.json", labels), ("queries.json", queries),
                        ("dataset_evidence.json", evidence), ("treatment_evidence.json", treatment),
                        ("packets.json", packets), ("input_support.json", supports), ("policy.json", policy)):
        write(bundle / name, value)
    write(bundle / "manifest.json", {p.name: sha(p) for p in bundle.iterdir() if p.is_file()})
    validate_bundle(bundle)
    return {"train": len(train), "test": len(queries), "agent_pilot": len(pilot),
            "prompt_chars": [min(len(x["selector"]) for x in packets), max(len(x["selector"]) for x in packets)]}


def validate_bundle(bundle=BUNDLE):
    from tools.outcome_prediction.wm_matched_evidence import extract
    from tools.outcome_prediction.wm_matched_fit import candidate_rows, neighborhood

    bundle = Path(bundle)
    old.previous.verify_files(bundle, read(bundle / "manifest.json"))
    policy = read(bundle / "policy.json")
    for name, expected in {**policy["sources"], **policy["linked_files"]}.items():
        if sha(name) != expected:
            raise ValueError("Changed frozen pilot dependency: " + name)
    if policy["selector_system"] != MATCH_SYSTEM or policy["predictor_system"] != DIRECT_SYSTEM or policy["variants"] != list(VARIANTS):
        raise ValueError("Changed policy")
    _, original_train, original_labels, original_queries, _, original_evidence, _ = prior.validate_bundle(prior.BUNDLE)
    train, labels, queries, evidence, treatment, packets, supports = (
        read(bundle / name) for name in ("train_inputs.json", "train_labels.json", "queries.json", "dataset_evidence.json",
                                       "treatment_evidence.json", "packets.json", "input_support.json"))
    if (train, labels, queries, evidence) != (original_train, original_labels, original_queries, original_evidence):
        raise ValueError("Changed clean membership/features/train labels")
    if treatment != {x["example_id"]: extract(x) for x in read(prior.SOURCE / "inputs.json")}:
        raise ValueError("Treatment evidence is not source-derived")
    pilot = [q for q in queries if q["reference_kind"] == "measured_parent"]
    if policy["pilot_ids"] != [q["example_id"] for q in pilot] or len(packets) != len(pilot):
        raise ValueError("Changed input-defined pilot")
    expected_support = [{"example_id": q["example_id"], "benchmark": q["benchmark"],
                         "neighborhoods": {v: neighborhood(task_pool(train, q), q, v, evidence=treatment) for v in VARIANTS}}
                        for q in queries]
    if supports != expected_support:
        raise ValueError("Changed outcome-blind support masks")
    for q, saved in zip(pilot, packets):
        pool = task_pool(train, q)
        candidates = candidate_rows(pool, q, evidence=treatment)
        prompt, aliases = packet(q, candidates, evidence, treatment, policy["feature_order"])
        bank = old.previous.select_bank(pool, q["benchmark"])
        fixed, _ = packet(q, bank, evidence, treatment, policy["feature_order"], labels=pool_labels(bank, labels),
                          weights=uniform_session_weights(bank), kind="fixed_16_session_diverse")
        expected = {"example_id": q["example_id"], "candidate_ids": [r["example_id"] for r in candidates],
                    "aliases": aliases, "selector": prompt, "selector_sha256": old.digest(prompt),
                    "fixed16": fixed, "fixed16_sha256": old.digest(fixed)}
        if saved != expected:
            raise ValueError("Changed matching/direct factual packet")
    return policy, train, labels, queries, evidence, treatment, packets


def numeric_forecasts(train, labels, queries, treatment):
    from tools.outcome_prediction.wm_matched_fit import fit_predict

    return [{"example_id": q["example_id"], "benchmark": q["benchmark"],
             "fits": {v: fit_predict(task_pool(train, q), pool_labels(task_pool(train, q), labels), q, v,
                                     evidence=treatment) for v in VARIANTS}} for q in queries]


def loso_forecasts(train, labels, treatment):
    results = []
    for group in sorted({r["cell_id"] for r in train}):
        fitting = [r for r in train if r["cell_id"] != group]
        held = [r for r in train if r["cell_id"] == group]
        results.extend(numeric_forecasts(fitting, pool_labels(fitting, labels), held, treatment))
    return sorted(results, key=lambda r: r["example_id"])


def run(bundle=BUNDLE, output=OUTPUT):
    from tools.outcome_prediction.wm_matched_fit import fit_from_neighborhood

    bundle, output = Path(bundle), Path(output)
    policy, train, labels, queries, evidence, treatment, packets = validate_bundle(bundle)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name in ("calls", "direct_prompts", "query_results"):
        (output / name).mkdir(mode=0o700)
    write(output / "run_policy.json", {"started_at": old.now(), "bundle_manifest_sha256": sha(bundle / "manifest.json"),
                                      "claude_version": subprocess.run(["claude", "--version"], capture_output=True, text=True, check=True).stdout.strip()})
    write(output / "numeric_predictions.json", numeric_forecasts(train, labels, queries, treatment))
    write(output / "train_loso_predictions.json", loso_forecasts(train, labels, treatment))
    write(output / "numeric_freeze.json", {name: sha(output / name) for name in ("numeric_predictions.json", "train_loso_predictions.json")})
    budget = recovery.CallBudget(policy["reported_cost_stop_usd"], policy["per_call_budget_usd"])

    def invoke(prompt, stage):
        invocation = {**policy, "system": policy["selector_system" if stage == "selector" else "predictor_system"]}
        return recovery._invoke(prompt, invocation, budget)

    probe = recovery._invoke('Return exactly {"ok":true}.',
                             {**policy, "system": "Return exactly the requested JSON object. No tools."}, budget)
    write(output / "probe.json", probe)
    probe_ok = not probe.get("error") and probe.get("response") == {"ok": True}
    packet_of = {p["example_id"]: p for p in packets}

    def attempt(query, prompt, stage, parse):
        key = old.key_for(query)
        attempts, error, parsed = 0, None, None
        if not probe_ok:
            return None, 0, "connectivity_probe_failed"
        for n in range(1, policy["max_attempts"] + 1):
            record = invoke(prompt, stage)
            attempts = n
            if record.get("response") is not None and not record.get("error"):
                try:
                    parsed, ignored = parse(record["response"])
                    record["ignored_fields"] = ignored
                except (ValueError, TypeError, KeyError) as exc:
                    record["error"] = "invalid_response: " + str(exc)
            record.update(example_id=query["example_id"], stage=stage, attempt=n, prompt_sha256=old.digest(prompt))
            write(output / "calls" / (key + f"__{stage}__attempt{n}.json"), record)
            error = record.get("error")
            if parsed is not None or not record.get("attempted") or budget.stopped:
                break
        return parsed, attempts, error

    def parse_forecast(response):
        return old.previous.parse_prediction(old.canonical(response)), {}

    def work(query):
        identity = query["example_id"]
        pool = task_pool(train, query)
        item = packet_of[identity]
        result = {"example_id": identity, "benchmark": query["benchmark"], "selection": None,
                  "agent_median": None, "agent_shrunk": None, "matched_llm": None, "fixed16_llm": None,
                  "selector_attempts": 0, "matched_attempts": 0, "fixed16_attempts": 0, "errors": {}}
        selection, n, error = attempt(query, item["selector"], "selector",
                                      lambda response: resolve(response, item["aliases"], pool, query, treatment))
        result["selector_attempts"] = n
        result["selection"] = selection
        if selection is None:
            result["errors"]["selector"] = error
        else:
            neighborhood = selection["neighborhood"]
            result["agent_median"] = fit_from_neighborhood(pool, pool_labels(pool, labels), query, neighborhood)
            result["agent_shrunk"] = fit_from_neighborhood(pool, pool_labels(pool, labels), query, neighborhood, shrink=True)
            if not neighborhood["support"]["supported"]:
                result["matched_llm"] = {"predicted_accuracy": query["parent_reference"], "predicted_delta": 0.0,
                                         "fallback": "insufficient_matching_support"}
            else:
                by_id = {r["example_id"]: r for r in pool}
                selected = [by_id[k] for k in neighborhood["selected_ids"]]
                prompt, _ = packet(query, selected, evidence, treatment, policy["feature_order"],
                                   labels=pool_labels(selected, labels), weights=neighborhood["effective_weights"],
                                   kind="agent_matched_training")
                write(output / "direct_prompts" / (old.key_for(query) + ".json"),
                      {"example_id": identity, "prompt": prompt, "prompt_sha256": old.digest(prompt)})
                pred, n, error = attempt(query, prompt, "matched", parse_forecast)
                result["matched_attempts"], result["matched_llm"] = n, pred
                if pred is None:
                    result["errors"]["matched"] = error
        pred, n, error = attempt(query, item["fixed16"], "fixed16", parse_forecast)
        result["fixed16_attempts"], result["fixed16_llm"] = n, pred
        if pred is None:
            result["errors"]["fixed16"] = error
        write(output / "query_results" / (old.key_for(query) + ".json"), result)
        return result

    results = []
    pilot = [q for q in queries if q["example_id"] in packet_of]
    with ThreadPoolExecutor(max_workers=policy["workers"]) as executor:
        pending = [executor.submit(work, q) for q in pilot]
        for future in as_completed(pending):
            item = future.result()
            results.append(item)
            support = item["selection"]["neighborhood"]["support"] if item["selection"] else None
            print(f"{len(results)}/{len(pilot)} {item['example_id']} support={support} errors={item['errors']} cost={budget.reported:.2f}", flush=True)
    write(output / "agent_predictions.json", sorted(results, key=lambda r: r["example_id"]))
    write(output / "prediction_freeze.json", {"created_at": old.now(), "reported_cost_usd": budget.reported,
          "charged_cost_usd": budget.charged, "files": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
          "note": "All numeric, selection and LLM outputs frozen before pilot test-error computation; prior holdout outcomes were explored previously."})
    return {"numeric_queries": len(queries), "agent_queries": len(results), "reported_cost_usd": budget.reported,
            "selection_success": sum(r["selection"] is not None for r in results),
            "direct_available": sum(r["matched_llm"] is not None for r in results)}


def verify_attempts(output, query, stage, count, prompt, policy, parse, expected):
    """Replay chronological first-valid responses, including failed raw attempts."""
    from tools.outcome_prediction.wm_local_agent_io import _decode, _events

    if type(count) is not int or not 0 <= count <= policy["max_attempts"]:
        raise ValueError("Invalid call count")
    accepted = None
    for i in range(1, count + 1):
        saved = read(output / "calls" / (old.key_for(query) + f"__{stage}__attempt{i}.json"))
        if (saved["example_id"] != query["example_id"] or saved["stage"] != stage
                or saved["attempt"] != i or saved["prompt_sha256"] != old.digest(prompt)):
            raise ValueError("Call/prompt/query misalignment")
        if not saved.get("attempted"):
            if not saved.get("error") or i != count:
                raise ValueError("Unattempted call has invalid chronology/status")
            continue
        if (saved["max_output_tokens"] != policy["max_output_tokens"] or saved["effort"] != policy["effort"]
                or saved["timeout_seconds"] != policy["timeout_seconds"]):
            raise ValueError("Changed actual execution allowance")
        events = _events(saved["stdout"])
        response, error, models, _ = _decode(events, policy["model_requested"], saved["returncode"], saved["stderr"])
        if events != saved["raw_events"] or models != saved["models_resolved"]:
            raise ValueError("Changed raw call record")
        execution_failed = (saved.get("error") == "request_timeout"
                            or str(saved.get("error", "")).startswith("invocation_error:"))
        if execution_failed:
            if saved["returncode"] is not None or saved["response"] is not None:
                raise ValueError("Malformed failed-execution record")
            # A timed-out process may already have printed a complete result.
            # Preserve the bytes, but never salvage it as a successful forecast.
            continue
        if response != saved["response"]:
            raise ValueError("Changed decoded response")
        if error:
            if not saved.get("error"):
                raise ValueError("Failed raw call marked successful")
            continue
        try:
            parsed, ignored = parse(response)
        except (ValueError, TypeError, KeyError):
            if not str(saved.get("error", "")).startswith("invalid_response:"):
                raise ValueError("Invalid semantic response was not recorded") from None
            continue
        if saved.get("error") or saved.get("ignored_fields") != ignored or accepted is not None or i != count:
            raise ValueError("Not first semantically valid response")
        old.validate_success_call(saved, policy)
        accepted = parsed
    if accepted != expected:
        raise ValueError("Saved forecast/selection differs from raw first valid response")


def metrics_for(rows, labels, forecasts):
    y = np.asarray([labels[q["example_id"]]["accuracy"] for q in rows])
    parent = np.asarray([q["parent_reference"] for q in rows])
    return {arm: {"accuracy": old.evaluate(rows, y, np.asarray(pred)),
                  "delta": old.evaluate(rows, y - parent, np.asarray(pred) - parent)}
            for arm, pred in forecasts.items()}


def compare(rows, labels, forecasts, arm, references):
    if not rows:
        return {}
    y = np.asarray([labels[q["example_id"]]["accuracy"] for q in rows])
    return {reference: old.compare_fixed_holdout(rows, y, np.asarray(forecasts[arm]), np.asarray(forecasts[reference]))
            for reference in references}


def historical_forecasts(queries):
    """Only called after this pilot's complete forecast freeze and replay."""
    fixed = {r["example_id"]: r for r in read(old.PREVIOUS / "predictions.json") if r["arm"] == "few_shot"}
    v1 = {r["example_id"]: r for r in read(prior.ROOT / "v1_recovery/predictions.json")}
    v2 = {r["example_id"]: r for r in read(prior.ROOT / "v2_completion/predictions.json")}
    regressors = read(old.REGRESSORS / "predictions.json")
    return {q["example_id"]: {
        "parent_unchanged": q["parent_reference"],
        "original_16shot": fixed[q["example_id"]]["predicted_accuracy"],
        "v1_agent_local": v1[q["example_id"]]["local_fit"]["predicted_accuracy"],
        "v2_agent_local": v2[q["example_id"]]["fits"]["agent_target"]["predicted_accuracy"],
        "global_train_selected": regressors["combined__" + q["benchmark"]][q["example_id"]]["train_selected"],
    } for q in queries}


def score(bundle=BUNDLE, output=OUTPUT):
    from tools.outcome_prediction.wm_matched_fit import fit_from_neighborhood

    bundle, output = Path(bundle), Path(output)
    if (output / "report.json").exists():
        raise FileExistsError("Preserve scored pilot")
    freeze = read(output / "prediction_freeze.json")
    old.previous.verify_files(output, freeze["files"])
    policy, train, train_labels, queries, evidence, treatment, packets = validate_bundle(bundle)
    if read(output / "run_policy.json")["bundle_manifest_sha256"] != sha(bundle / "manifest.json"):
        raise ValueError("Changed run/bundle link")
    numeric, cv, agent = (read(output / name) for name in ("numeric_predictions.json", "train_loso_predictions.json", "agent_predictions.json"))
    if numeric != numeric_forecasts(train, train_labels, queries, treatment) or cv != loso_forecasts(train, train_labels, treatment):
        raise ValueError("Numerical or full-procedure LOSO replay failed")
    by_agent = {r["example_id"]: r for r in agent}
    if len(by_agent) != len(agent) or set(by_agent) != set(policy["pilot_ids"]):
        raise ValueError("Changed agent pilot membership")
    packet_of = {p["example_id"]: p for p in packets}
    for q in queries:
        if q["example_id"] not in by_agent:
            continue
        saved = by_agent[q["example_id"]]
        if saved != read(output / "query_results" / (old.key_for(q) + ".json")):
            raise ValueError("Changed query result")
        pool = task_pool(train, q)
        item = packet_of[q["example_id"]]
        verify_attempts(output, q, "selector", saved["selector_attempts"], item["selector"], policy,
                        lambda response, aliases=item["aliases"], rows=pool, query=q:
                        resolve(response, aliases, rows, query, treatment), saved["selection"])
        if saved["selection"] is None:
            if any(saved[k] is not None for k in ("agent_median", "agent_shrunk", "matched_llm")) or saved["matched_attempts"]:
                raise ValueError("Missing selection cannot become forecast/fallback")
        else:
            n = saved["selection"]["neighborhood"]
            for key, shrink in (("agent_median", False), ("agent_shrunk", True)):
                if saved[key] != fit_from_neighborhood(pool, pool_labels(pool, train_labels), q, n, shrink=shrink):
                    raise ValueError("Agent numerical prediction failed replay")
            if not n["support"]["supported"]:
                expected = {"predicted_accuracy": q["parent_reference"], "predicted_delta": 0.0,
                            "fallback": "insufficient_matching_support"}
                if saved["matched_llm"] != expected or saved["matched_attempts"]:
                    raise ValueError("Unsupported matching/direct fallback differs")
            else:
                by_id = {r["example_id"]: r for r in pool}
                selected = [by_id[k] for k in n["selected_ids"]]
                prompt, _ = packet(q, selected, evidence, treatment, policy["feature_order"],
                                   labels=pool_labels(selected, train_labels), weights=n["effective_weights"], kind="agent_matched_training")
                if read(output / "direct_prompts" / (old.key_for(q) + ".json")) != {
                        "example_id": q["example_id"], "prompt": prompt, "prompt_sha256": old.digest(prompt)}:
                    raise ValueError("Matched direct factual context/weights changed")
                verify_attempts(output, q, "matched", saved["matched_attempts"], prompt, policy,
                                lambda response: (old.previous.parse_prediction(old.canonical(response)), {}), saved["matched_llm"])
        verify_attempts(output, q, "fixed16", saved["fixed16_attempts"], item["fixed16"], policy,
                        lambda response: (old.previous.parse_prediction(old.canonical(response)), {}), saved["fixed16_llm"])
    # Pilot outcomes/older forecasts are decoded only after all forecast checks.
    targets = read(old.TRAINING / "labels.json")
    histories = historical_forecasts(queries)
    by_num = {r["example_id"]: r for r in numeric}
    by_cv = {r["example_id"]: r for r in cv}
    report = {"scored_at": old.now(), "prediction_frozen_at": freeze["created_at"],
              "reported_cost_usd": freeze["reported_cost_usd"], "limitations": policy["limitations"],
              "numeric_test": {}, "train_loso": {}, "agent_pilot": {}}
    for benchmark in old.previous.TASKS:
        all_q = [q for q in queries if q["benchmark"] == benchmark]

        def predictions(rows, with_agent=False):
            values = {}
            for q in rows:
                identity = q["example_id"]
                row = {v: by_num[identity]["fits"][v]["predicted_accuracy"] for v in VARIANTS}
                row.update(histories[identity])
                if with_agent:
                    row.update({key: by_agent[identity][key]["predicted_accuracy"]
                                for key in ("agent_median", "agent_shrunk", "matched_llm", "fixed16_llm")})
                for key, value in row.items():
                    values.setdefault(key, []).append(value)
            return values

        subsets = {"all": all_q,
                   **{kind: [q for q in all_q if q["reference_kind"] == kind] for kind in ("published_base", "measured_parent")},
                   "common_rule_supported": [q for q in all_q if all(by_num[q["example_id"]]["fits"][v]["neighborhood"]["support"]["supported"]
                                                                 for v in ("state_only", "state_treatment_soft"))]}
        report["numeric_test"][benchmark] = {
            "support": {v: {"supported": sum(by_num[q["example_id"]]["fits"][v]["neighborhood"]["support"]["supported"] for q in all_q),
                             "intended": len(all_q)} for v in VARIANTS},
            "subsets": {name: {"ids": [q["example_id"] for q in rows], "metrics": metrics_for(rows, targets, predictions(rows)),
                               "comparisons": compare(rows, targets, predictions(rows), "state_treatment_soft",
                                                      ("all_rows", "state_only", "state_tag", "original_16shot", "v2_agent_local"))}
                        for name, rows in subsets.items()}}
        train_q = [q for q in train if q["benchmark"] == benchmark]
        report["train_loso"][benchmark] = {}
        for kind in ("all", "published_base", "measured_parent"):
            rows = [q for q in train_q if kind == "all" or q["reference_kind"] == kind]
            forecasts = {v: [by_cv[q["example_id"]]["fits"][v]["predicted_accuracy"] for q in rows] for v in VARIANTS}
            forecasts["parent_unchanged"] = [q["parent_reference"] for q in rows]
            report["train_loso"][benchmark][kind] = {
                "metrics": metrics_for(rows, train_labels, forecasts),
                "supported": {v: sum(by_cv[q["example_id"]]["fits"][v]["neighborhood"]["support"]["supported"] for q in rows) for v in VARIANTS}}
        intended = [q for q in all_q if q["example_id"] in by_agent]
        common = [q for q in intended if all(by_agent[q["example_id"]][key] is not None
                                           for key in ("agent_median", "agent_shrunk", "matched_llm", "fixed16_llm"))]
        supported = [q for q in common if by_agent[q["example_id"]]["selection"]["neighborhood"]["support"]["supported"]]
        pilot_report = {"intended": len(intended), "common_available": len(common), "common_supported": len(supported),
                        "coverage": {key: sum(by_agent[q["example_id"]][key] is not None for q in intended)
                                     for key in ("selection", "agent_median", "agent_shrunk", "matched_llm", "fixed16_llm")},
                        "subsets": {}}
        for name, rows in (("common_with_fallback", common), ("common_supported", supported)):
            forecasts = predictions(rows, with_agent=True)
            pilot_report["subsets"][name] = {"ids": [q["example_id"] for q in rows], "metrics": metrics_for(rows, targets, forecasts),
                "comparisons": compare(rows, targets, forecasts, "agent_median",
                                       ("state_treatment_soft", "agent_shrunk", "matched_llm", "fixed16_llm", "original_16shot", "v2_agent_local"))}
        report["agent_pilot"][benchmark] = pilot_report
    write(output / "report.json", report)
    write(output / "manifest.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    return {"numeric": {b: {arm: round(m["accuracy"]["mae"] * 100, 3) for arm, m in v["subsets"]["all"]["metrics"].items()}
                        for b, v in report["numeric_test"].items()},
            "agent": {b: {"n": v["common_available"], "supported": v["common_supported"],
                          "mae_pp": {arm: round(m["accuracy"]["mae"] * 100, 3)
                                     for arm, m in v["subsets"]["common_with_fallback"]["metrics"].items()}}
                      for b, v in report["agent_pilot"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = prepare(args.bundle) if args.action == "prepare" else (
        run(args.bundle, args.output) if args.action == "run" else score(args.bundle, args.output))
    print(old.canonical(value))


if __name__ == "__main__":
    main()
