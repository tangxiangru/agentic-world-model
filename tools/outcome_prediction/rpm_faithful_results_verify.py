"""Audit saved rich-RPM responses and independently recompute published metrics.

Makes no model calls and never changes inputs, original responses, decoded
responses, learned predictions, or metrics. The only output is
results_verification.json. Partial coverage is expected until --require-complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from tools.outcome_prediction.rpm_judge import digest
from tools.outcome_prediction.rpm_paper_prompt import decode_boxed_response

REFERENCE_METHODS = (
    "later_recipe",
    "frozen/within_run",
    "frozen/cross_run",
    "sibling_train/exploratory_recipe_numeric_forest",
    "contextual/exploratory_recipe_numeric_contextual_forest",
)
TOKEN_FIELDS = (
    "inputTokens",
    "outputTokens",
    "cacheCreationInputTokens",
    "cacheReadInputTokens",
    "thinkingTokens",
    "webSearchRequests",
)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_summary(records, method):
    groups = defaultdict(list)
    for row in records:
        winner_a = row["y_a"] > row["y_b"]
        forced = row.get("choices", {}).get(method)
        probability = row["probabilities"][method]
        if forced is not None:
            credit = float(forced == winner_a)
        elif abs(probability - 0.5) < 1e-12:
            credit = 0.5
        else:
            credit = float((probability > 0.5) == winner_a)
        regret = (1 - credit) * abs(row["y_a"] - row["y_b"])
        groups[row["cell_id"]].append((credit, regret))
    all_values = [value for group in groups.values() for value in group]
    return {
        "pairs": len(records),
        "cells": len(groups),
        "micro_accuracy": mean(x[0] for x in all_values),
        "macro_accuracy": mean(mean(x[0] for x in group) for group in groups.values()),
        "micro_regret": mean(x[1] for x in all_values),
        "macro_regret": mean(mean(x[1] for x in group) for group in groups.values()),
    }


def verify(root, require_complete=False):
    destination = root / "results_verification.json"
    previous = json.loads(destination.read_text()) if destination.exists() else {}
    jobs = {j["id"]: j for j in json.loads((root / "jobs.json").read_text())}
    label_list = json.loads((root / "hidden_labels.json").read_text())
    labels = {r["id"]: r for r in label_list}
    protocol = json.loads((root / "protocol.json").read_text())
    original_paths = sorted((root / "outputs").glob("*.json"))
    originals = {p.stem: json.loads(p.read_text()) for p in original_paths}
    derived_paths = sorted((root / "decoded_outputs").glob("*.json"))
    derived = {p.stem: json.loads(p.read_text()) for p in derived_paths}
    output_hashes = {str(p): file_hash(p) for p in original_paths}
    errors, warnings, response_audits, job_hashes = [], [], [], []
    counts = Counter()
    model_usage = defaultdict(Counter)
    aliases = defaultdict(set)
    accepted, selected, unknown_cost_ids = {}, {}, []

    def check(test, identity, reason):
        if not test:
            errors.append({"id": identity, "reason": reason})

    def close(actual, expected, identity, reason):
        check(
            isinstance(actual, (int, float))
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            identity,
            reason,
        )

    for path, original_hash in previous.get("output_file_sha256", {}).items():
        check(path in output_hashes, path, "previously observed original output disappeared")
        if path in output_hashes:
            check(
                output_hashes[path] == original_hash,
                path,
                "original output changed since prior audit",
            )
    for identity, job in jobs.items():
        prompt = Path(job["prompt_path"]).read_text()
        packet = json.loads((root / "inputs" / (identity + ".json")).read_text())
        check(digest(prompt) == job["prompt_sha256"], identity, "prompt hash mismatch")
        check(digest(packet) == job["input_sha256"], identity, "input hash mismatch")
        job_hashes.append(
            {"id": identity, "prompt_sha256": digest(prompt), "input_sha256": digest(packet)}
        )
    check(set(derived) <= set(originals), "derived_outputs", "derived response lacks original")
    for identity, original in originals.items():
        if identity not in jobs:
            check(False, identity, "response is outside frozen jobs")
            continue
        job = jobs[identity]
        counts["attempted_jobs"] += 1
        counts["attempted_main" if identity in labels else "attempted_swap"] += 1
        check(
            all(original.get(k) == v for k, v in job.items()),
            identity,
            "saved response/job metadata mismatch",
        )
        check(
            original.get("model_requested") == protocol["model"],
            identity,
            "requested model changed",
        )
        raw = original.get("raw_response")
        decoded, decoder_error = None, None
        try:
            decoded = decode_boxed_response(raw, swapped=job["swapped"])
        except (ValueError, KeyError, TypeError) as error:
            decoder_error = type(error).__name__ + ": " + str(error)
        accepted[identity] = decoded
        if original["valid"]:
            check(decoded is not None, identity, "original valid answer rejected by fixed decoder")
            if decoded:
                check(
                    all(original.get(k) == v for k, v in decoded.items()),
                    identity,
                    "original decoded fields mismatch",
                )
        if identity in derived:
            result = derived[identity]
            counts["derived_outputs"] += 1
            check(result.get("raw_response") == raw, identity, "derived raw response changed")
            check(
                result.get("original_decode_valid") == original["valid"],
                identity,
                "original decode status not preserved",
            )
            for key in set(original) - {
                "valid",
                "error",
                "choice_a",
                "displayed_choice",
                "rationale",
            }:
                check(
                    result.get(key) == original[key],
                    identity,
                    "derived nondecoder field changed: " + key,
                )
            check(
                result["valid"] == (decoded is not None),
                identity,
                "derived valid status differs from decoder",
            )
            if decoded:
                check(
                    all(result.get(k) == v for k, v in decoded.items()),
                    identity,
                    "derived decoded fields mismatch",
                )
                counts["repaired_original_invalid"] += not original["valid"]
        else:
            result = original
            if decoded and not original["valid"]:
                counts["pending_derived_decodes"] += 1
        selected[identity] = result
        if result["valid"]:
            check(decoded is not None, identity, "reported answer rejected by fixed decoder")
            check(
                result["choice_a"] == ((result["displayed_choice"] == "A") != job["swapped"]),
                identity,
                "choice orientation mismatch",
            )
            counts["valid_main" if identity in labels else "valid_swap"] += 1
        else:
            counts["invalid_main" if identity in labels else "invalid_swap"] += 1
        if "cost_usd" not in original:
            unknown_cost_ids.append(identity)
        audit = {
            "id": identity,
            "original_valid": original["valid"],
            "derived_present": identity in derived,
            "reported_valid": result["valid"],
            "fixed_decoder_valid": decoded is not None,
            "decoder_error": decoder_error,
            "raw_response_present": raw is not None,
            "raw_response_sha256": digest(raw) if raw is not None else None,
        }
        if raw is None:
            counts["responses_without_raw_response"] += 1
            if "TimeoutExpired" in original.get("error", ""):
                counts["timeout_without_partial_response"] += 1
                warnings.append(
                    {
                        "id": identity,
                        "reason": "Timeout has no preserved raw/partial response or reported charge; cost is unknown, not zero.",
                    }
                )
            check(not result["valid"], identity, "valid answer without raw response")
        else:
            counts["raw_responses"] += 1
            usage = raw.get("modelUsage", {})
            canonical_models = {entry.get("canonicalModel", key) for key, entry in usage.items()}
            check(
                protocol["model"] in canonical_models,
                identity,
                "requested canonical primary model missing from usage",
            )
            unexpected = canonical_models - {protocol["model"], "claude-haiku-4-5"}
            check(not unexpected, identity, "unexpected model usage: " + str(sorted(unexpected)))
            audit["canonical_models"] = sorted(canonical_models)
            subtotal = 0
            for key, entry in usage.items():
                canonical = entry.get("canonicalModel", key)
                aliases[canonical].add(key)
                model_usage[canonical]["responses"] += 1
                model_usage[canonical]["cost_usd"] += entry.get("costUSD", 0)
                subtotal += entry.get("costUSD", 0)
                for field in TOKEN_FIELDS:
                    model_usage[canonical][field] += entry.get(field, 0)
            close(
                original.get("cost_usd"),
                raw.get("total_cost_usd", 0),
                identity,
                "saved cost differs from raw total",
            )
            close(
                subtotal,
                raw.get("total_cost_usd", 0),
                identity,
                "model usage costs do not sum to raw total",
            )
            if len(canonical_models) > 1:
                counts["raw_responses_with_auxiliary_model"] += 1
        response_audits.append(audit)

    learned_path = root / "rich_learned.json"
    folds_path = root / "rich_learned_folds.json"
    learned = {r["id"]: r for r in json.loads(learned_path.read_text())}
    folds = json.loads(folds_path.read_text())
    check(set(learned) == set(labels), "learned", "prediction cohort mismatch")
    expected_methods = set(protocol["learned_methods_frozen"]) - {"shared"}
    for identity, prediction in learned.items():
        check(prediction["fold"] == labels[identity]["fold"], identity, "prediction fold mismatch")
        check(
            set(prediction["probabilities"]) == expected_methods,
            identity,
            "learned method specification mismatch",
        )
        check(
            all(math.isfinite(v) and 0 <= v <= 1 for v in prediction["probabilities"].values()),
            identity,
            "invalid learned probability",
        )
    seen_test = Counter()
    for fold in folds:
        expected_train = {p["id"] for p in label_list if p["fold"] != fold["fold"]}
        expected_test = set(labels) - expected_train
        check(
            set(fold["train_pairs"]) == expected_train,
            str(fold["fold"]),
            "fold training bank mismatch",
        )
        check(
            set(fold["test_pairs"]) == expected_test, str(fold["fold"]), "fold test bank mismatch"
        )
        train_cells = {labels[p]["cell_id"] for p in fold["train_pairs"]}
        test_cells = {labels[p]["cell_id"] for p in fold["test_pairs"]}
        check(
            not (train_cells & test_cells), str(fold["fold"]), "held-out cell leaked into training"
        )
        check(
            set(fold["train_cells"]) == train_cells,
            str(fold["fold"]),
            "stored train cell audit mismatch",
        )
        seen_test.update(fold["test_pairs"])
    check(
        set(seen_test) == set(labels) and all(n == 1 for n in seen_test.values()),
        "folds",
        "test pairs not covered exactly once",
    )
    counts["learned_predictions"] = len(learned)
    counts["verified_folds"] = len(folds)

    missing_jobs = sorted(set(jobs) - set(originals))
    if require_complete:
        check(not missing_jobs, "coverage", "frozen jobs remain unattempted")
        check(
            not counts["pending_derived_decodes"], "decoding", "recoverable answers not yet derived"
        )
    metric_result = {"state": "not yet available; performance not computed"}
    metrics_path = root / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        missing_main = sorted(set(labels) - set(selected))
        attempted_main = [identity for identity in labels if identity in selected]
        failed_main = sorted(
            identity for identity in attempted_main if not selected[identity]["valid"]
        )
        fresh = (
            sorted(metrics["missing_pairs"]) == missing_main
            and sorted(metrics["failed_pairs"]) == failed_main
            and metrics["attempted_main_pairs"] == len(attempted_main)
        )
        if not fresh and not require_complete:
            metric_result = {
                "state": "metrics stale relative to currently available outputs; not compared"
            }
        else:
            check(fresh, "metrics", "coverage metrics stale or incorrect")
            old_path = root.parent / "comparison_pairs.jsonl"
            old = {r["id"]: r for r in map(json.loads, old_path.read_text().splitlines())}
            records = []
            for label in label_list:
                identity = label["id"]
                if identity not in selected or not selected[identity]["valid"]:
                    continue
                result = selected[identity]
                row = {
                    **label,
                    "probabilities": {"chance": 0.5, "rpm_rich": float(result["choice_a"])},
                    "choices": {"rpm_rich": result["choice_a"]},
                }
                row["probabilities"].update(learned[identity]["probabilities"])
                for method in REFERENCE_METHODS:
                    if method in old.get(identity, {}).get("probabilities", {}):
                        row["probabilities"][method] = old[identity]["probabilities"][method]
                        if method in old[identity].get("choices", {}):
                            row["choices"][method] = old[identity]["choices"][method]
                close(
                    label["gap"], abs(label["y_a"] - label["y_b"]), identity, "label gap mismatch"
                )
                records.append(row)
            groups = {
                "all_audited": records,
                "unredacted_plans": [r for r in records if r.get("full_plan_strict_eligible")],
                "known_nonbase_parent": [r for r in records if r.get("has_nonbase_parent")],
                "gap_02": [r for r in records if r["gap"] >= 0.02],
            }
            summaries = {}
            for group_name, group in groups.items():
                if not group:
                    continue
                methods = set.intersection(*(set(r["probabilities"]) for r in group))
                summaries[group_name] = {
                    method: independent_summary(group, method) for method in sorted(methods)
                }
                for method, expected in summaries[group_name].items():
                    actual = metrics["summary"].get(group_name, {}).get(method, {})
                    for key, value in expected.items():
                        close(
                            actual.get(key),
                            value,
                            group_name + "/" + method,
                            "summary mismatch: " + key,
                        )
                    check(
                        not any("brier" in key or "log_loss" in key for key in actual),
                        method,
                        "hard-choice probability loss reported",
                    )
                    if method == "rpm_rich":
                        continue
                    comparison = (
                        metrics["comparison_to_rpm_rich"].get(group_name, {}).get(method, {})
                    )
                    reference = summaries[group_name]["rpm_rich"]
                    for weight in ("macro", "micro"):
                        close(
                            comparison.get(weight + "_accuracy_gain"),
                            expected[weight + "_accuracy"] - reference[weight + "_accuracy"],
                            method,
                            "accuracy difference mismatch",
                        )
                        close(
                            comparison.get(weight + "_regret_reduction"),
                            reference[weight + "_regret"] - expected[weight + "_regret"],
                            method,
                            "regret difference mismatch",
                        )
            close(metrics["prepared_pairs"], len(labels), "metrics", "prepared count mismatch")
            close(metrics["valid_pairs"], len(records), "metrics", "valid count mismatch")
            close(
                metrics["reported_cost_usd"],
                sum(selected[p].get("cost_usd", 0) for p in attempted_main),
                "metrics",
                "main reported cost mismatch",
            )
            close(
                metrics["reported_cost_with_swaps_usd"],
                sum(r.get("cost_usd", 0) for r in selected.values()),
                "metrics",
                "all reported cost mismatch",
            )
            close(
                metrics.get("unknown_cost_calls"),
                len(unknown_cost_ids),
                "metrics",
                "unknown-cost count mismatch",
            )
            swaps = [r for identity, r in selected.items() if identity not in labels]
            comparable = [
                r
                for r in swaps
                if r["valid"] and r["swap_of"] in selected and selected[r["swap_of"]]["valid"]
            ]
            expected_swaps = {
                "attempted": len(swaps),
                "valid_paired": len(comparable),
                "consistent": sum(
                    r["choice_a"] == selected[r["swap_of"]]["choice_a"] for r in comparable
                ),
            }
            check(metrics["swap_checks"] == expected_swaps, "metrics", "swap consistency mismatch")
            saved_records = json.loads((root / "comparison_records.json").read_text())
            check(saved_records == records, "metrics", "comparison records mismatch")
            metric_result = {
                "state": "independently recomputed",
                "summary": summaries,
                "swap_checks": expected_swaps,
                "confidence_intervals": "Not independently recomputed; point metrics and pairwise point differences verified.",
            }
    elif require_complete:
        check(False, "metrics", "final metrics absent")
    return {
        "schema": "rpm-faithful-results-verification-v1",
        "at_utc": datetime.now(UTC).isoformat(),
        "passed": not errors,
        "complete_output_coverage": not missing_jobs,
        "counts": dict(counts),
        "prepared_main": len(labels),
        "prepared_jobs": len(jobs),
        "missing_jobs": missing_jobs,
        "errors": errors,
        "warnings": warnings,
        "reported_total_cost_usd": sum(r.get("cost_usd", 0) for r in originals.values()),
        "unknown_cost_job_ids": unknown_cost_ids,
        "model_usage": {key: dict(value) for key, value in sorted(model_usage.items())},
        "model_aliases": {key: sorted(value) for key, value in aliases.items()},
        "usage_policy": "Aggregate modelUsage by canonical model. Top-level usage may omit CLI auxiliary Haiku usage. Reserved cost is not asserted to be actual spending.",
        "responses": response_audits,
        "metrics": metric_result,
        "protocol_sha256": digest(protocol),
        "jobs": job_hashes,
        "output_file_sha256": output_hashes,
        "derived_file_sha256": {str(p): file_hash(p) for p in derived_paths},
        "source_file_sha256": {
            str(p): file_hash(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("rpm_paper_prompt.py"),
                Path(__file__).with_name("rpm_faithful.py"),
            )
        },
        "learned_file_sha256": {str(p): file_hash(p) for p in (learned_path, folds_path)},
        "preservation_scope": "Original output hashes are compared with the prior verification run when available; no claim is made about content before the first observed hash.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/analysis/rpm/faithful"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = verify(args.root, args.require_complete)
    (args.root / "results_verification.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "complete_output_coverage",
                    "counts",
                    "errors",
                    "unknown_cost_job_ids",
                )
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
