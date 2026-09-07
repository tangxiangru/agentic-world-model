"""Verify RPM v2 inputs and available results; only the verification report is written.

This checks provenance and information boundaries, not predictive performance.
It makes no model calls and does not modify source data, prompts, or predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import rpm_judge as judge
except ImportError:
    import rpm_judge as judge


LEARNED_DIRS = ("learned", "learned_contextual", "learned_siblings")
FALSE_CONTEMPORANEOUS = (
    ("r0-06/exp-03", "r0-06/exp-04", "2026-09-04T05:48:36Z", "80.3%"),
    ("r0-18/exp-03", "r0-18/exp-04", "2026-09-04T07:14:56Z", "60.7%"),
    ("r0-19/exp-03", "r0-19/exp-04", "2026-09-04T03:22:05Z", "22%"),
    ("r0-25/exp-02", "r0-25/exp-03", "2026-09-04T01:09:00Z", "0.027"),
    ("r0-30/exp-05", "r0-30/exp-06", "2026-09-04T06:33:55Z", "0.4833"),
)
RECIPE_KEYS = {
    "base_model",
    "configuration",
    "data",
    "framework",
    "hyperparameters",
    "merge",
    "method",
    "parent_kind",
    "peft",
    "planned_hours",
}
FORBIDDEN_KEYS = {
    "y",
    "accuracy",
    "official_accuracy",
    "observed_accuracy",
    "result",
    "conclusion",
    "outcome",
    "example_id",
    "cell_id",
    "card_id",
    "scientist_model",
    "local_accuracy",
    "parent_accuracy",
    "comparator_accuracy",
    "plan_text",
}


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_id(a, b):
    return "pair-" + hashlib.sha256((a + "|" + b).encode()).hexdigest()[:12]


def siblings(a, b, complete=True):
    return (
        a["cell_id"] == b["cell_id"]
        and all(r["recipe"]["method"] != "merge" and len(r["parent_ids"]) <= 1 for r in (a, b))
        and sorted(a["parent_ids"]) == sorted(b["parent_ids"])
        and (not complete or a["lineage_complete"] and b["lineage_complete"])
    )


def independent_score(card):
    values = []
    for measurement in (card.get("result") or {}).get("measurements") or []:
        if not isinstance(measurement, dict):
            continue
        names = " ".join(str(measurement.get(k) or "") for k in ("name", "metric")).lower()
        if not any(word in names for word in ("accuracy", "gsm8k")):
            continue
        try:
            value = float(measurement["value"])
            n = float(measurement.get("n") or 0)
        except (ValueError, TypeError, KeyError):
            continue
        if 1 < value <= 100:
            value /= 100
        if math.isfinite(value) and math.isfinite(n) and 0 <= value <= 1:
            values.append((n, value))
    if not values:
        return None
    maximum = max(n for n, _ in values)
    scores = {value for n, value in values if n == maximum}
    return next(iter(scores)) if len(scores) == 1 else None


def independent_history(rows, records, a, b):
    cutoff = min(a["first_submitted_at"], b["first_submitted_at"])
    blocked = {a["card_id"], b["card_id"]}
    result = []
    for row in rows:
        if row["cell_id"] != a["cell_id"] or row["first_submitted_at"] >= cutoff:
            continue
        if row["card_id"] in blocked or blocked.intersection(x["card_id"] for x in row["lineage"]):
            continue
        known = [
            r
            for r in records[row["example_id"]]
            if r["at"] < cutoff and independent_score(r["card"]) is not None
        ]
        if known:
            latest = max(known, key=lambda r: r["at"])
            result.append(
                {
                    "recipe_sequence": [s["recipe"] for s in row["lineage"][:-1]]
                    + [judge.canonical(latest["card"])[0]],
                    "observed_accuracy": independent_score(latest["card"]),
                    "metric_scope": "scientist-reported local evaluation; subset/protocol may vary",
                }
            )
    return result


def all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from all_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from all_keys(value)


def expand_catalog(compact):
    if "historical_recipe_catalog" not in compact:
        return compact
    restored = {
        k: v
        for k, v in compact.items()
        if k
        not in {
            "historical_recipe_catalog",
            "historical_recipe_encoding",
            "historical_training_experiments",
        }
    }
    restored["historical_training_experiments"] = [
        {
            **experiment,
            "recipe_sequence": [
                compact["historical_recipe_catalog"][key] for key in experiment["recipe_sequence"]
            ],
        }
        for experiment in compact["historical_training_experiments"]
    ]
    return restored


def verify(args):
    errors, checks = [], Counter()

    def check(condition, name, context=""):
        checks[name] += 1
        if not condition:
            errors.append({"check": name, "context": context})

    raw_rows = jsonl(args.examples)
    rows = [r for r in raw_rows if r["eligible"] and r["y"] is not None]
    by_id = {r["example_id"]: r for r in raw_rows}
    eligible_ids = {r["example_id"] for r in rows}
    check(len(eligible_ids) == len(rows) == 146, "eligible_cohort")
    all_siblings, primary = [], {}
    for a, b in itertools.combinations(sorted(rows, key=lambda r: r["example_id"]), 2):
        gap = abs(a["y"] - b["y"])
        if siblings(a, b, complete=False):
            all_siblings.append((a, b, gap))
        if siblings(a, b) and gap >= 0.01:
            primary[pair_id(a["example_id"], b["example_id"])] = (a, b)
    check(len(primary) == 89, "primary_89_pairs")
    check(
        sum(abs(a["y"] - b["y"]) >= 0.02 for a, b in primary.values()) == 77, "sensitivity_77_pairs"
    )

    root = args.rpm_root
    judge_dir = root / "judge"
    protocol = read(judge_dir / "protocol.json")
    check(protocol.get("version") == 2, "protocol_v2")
    check(protocol["system_prompt_sha256"] == judge.digest(judge.SYSTEM), "system_prompt_hash")
    check(protocol.get("pairs") == len(primary), "protocol_pair_count")
    check("no fitted ensemble weights" in protocol.get("fixed_hybrid", ""), "fixed_blend_declared")
    folds = read(judge_dir / "folds.json")
    check(len(folds) == 8, "eight_outer_folds")
    fold_for, banks, test_coverage = {}, {}, Counter()
    for fold in folds:
        tr, te = set(fold["train_ids"]), set(fold["test_ids"])
        fid = fold["fold"]
        check(
            not tr.intersection(te) and tr.union(te) == eligible_ids, "fold_partitions_cohort", fid
        )
        check(
            not {by_id[i]["cell_id"] for i in tr}.intersection(by_id[i]["cell_id"] for i in te),
            "fold_holds_out_whole_cells",
            fid,
        )
        test_coverage.update(te)
        for eid in te:
            fold_for[eid] = fid
        banks[fid] = sorted(
            [by_id[i] for i in tr], key=lambda r: judge.digest(["training-order", r["example_id"]])
        )
    check(
        set(test_coverage) == eligible_ids and set(test_coverage.values()) == {1},
        "test_fold_coverage",
    )

    learned_coverage = {}
    for name in LEARNED_DIRS:
        assignments = read(root / name / "fold_assignments.json")
        check(len(assignments) == 8, "learned_eight_folds", name)
        amap = {f["fold"]: f for f in assignments}
        for fold in folds:
            learned = amap.get(fold["fold"], {})
            check(
                all(
                    set(learned.get(key, [])) == set(fold[key]) for key in ("train_ids", "test_ids")
                ),
                "identical_learned_fold_bank",
                f"{name}/{fold['fold']}",
            )
            frozen = read(root / name / f"fold-{fold['fold']:02d}-frozen.json")
            check(frozen == learned, "frozen_fold_equals_assignments", f"{name}/{fold['fold']}")
        predictions = jsonl(root / name / "pair_predictions.jsonl")
        covered = set()
        for prediction in predictions:
            aid, bid = prediction["a_id"], prediction["b_id"]
            check(
                fold_for[aid] == fold_for[bid] == prediction["fold"],
                "learned_prediction_fold",
                name,
            )
            if pair_id(*sorted((aid, bid))) in primary:
                covered.add(pair_id(*sorted((aid, bid))))
        check(covered == set(primary), "learned_covers_primary_pairs", name)
        learned_coverage[name] = {
            "all_prediction_pairs": len(predictions),
            "primary_pairs": len(covered),
        }

    records = {
        row["example_id"]: [
            read(path)
            for path in sorted((args.raw_root / row["source_card"]).parent.glob("record-*.json"))
        ]
        for row in raw_rows
    }
    history = {}
    for key, (a, b) in primary.items():
        expected = independent_history(raw_rows, records, a, b)
        check(
            expected == judge.observable_history(raw_rows, args.raw_root, a, b),
            "independent_raw_history",
            key,
        )
        history[key] = expected
    history_counts = Counter(len(h) for h in history.values())
    check(sum(bool(h) for h in history.values()) == 46, "46_pairs_with_history")
    check(dict(history_counts) == {0: 43, 1: 24, 2: 13, 3: 4, 4: 3, 5: 2}, "history_histogram")

    labels = read(judge_dir / "hidden_labels.json")
    check({r["id"] for r in labels} == set(primary) and len(labels) == 89, "hidden_label_cohort")
    for label in labels:
        a, b = primary[label["id"]]
        check(
            (label["a_id"], label["b_id"]) == (a["example_id"], b["example_id"]),
            "label_pair_identity",
            label["id"],
        )
        check(
            (label["y_a"], label["y_b"]) == (a["y"], b["y"]),
            "label_matches_official_examples",
            label["id"],
        )
        check(
            label["history_count"] == len(history[label["id"]]),
            "label_history_metadata",
            label["id"],
        )
    all_jobs = read(judge_dir / "jobs.json")
    jobs = [j for j in all_jobs if j["arm"] in {"within_run", "cross_run"}]
    expected_jobs = {(key, arm) for key in primary for arm in ("within_run", "cross_run")}
    check(
        {(j["id"], j["arm"]) for j in jobs} == expected_jobs and len(jobs) == 178, "178_unique_jobs"
    )
    outputs = Counter()
    models, incomplete_outputs = set(), []
    for job in all_jobs:
        key, arm = job["id"], job["arm"]
        a, b = primary[key]
        swap = int(hashlib.sha256((key + ":order").encode()).hexdigest(), 16) % 2 == 1
        base_arm = arm.removesuffix("_swap")
        if arm.endswith("_swap"):
            swap = not swap
        check(job["swapped"] == swap, "deterministic_candidate_order", key)
        payload = read(Path(job["input"]))
        check(judge.digest(payload) == job["input_sha256"], "input_hash", f"{arm}/{key}")
        bank = banks[fold_for[a["example_id"]]]
        check(
            not {a["cell_id"], b["cell_id"]}.intersection(r["cell_id"] for r in bank),
            "zero_target_cell_in_bank",
            key,
        )
        expected = judge.build_payload(a, b, bank, history[key], base_arm, swap)
        check(payload == expected, "payload_matches_candidates_bank_history", f"{arm}/{key}")
        if base_arm == "cross_run":
            groups = payload["historical_training_run_groups"]["groups"]
            check(
                sorted(i for group in groups for i in group) == list(range(len(bank))),
                "anonymous_groups_partition_bank",
                key,
            )
            check(
                all(len({bank[i]["cell_id"] for i in group}) == 1 for group in groups),
                "anonymous_groups_match_real_runs",
                key,
            )
        for side in ("candidate_A", "candidate_B"):
            candidate = payload[side]
            check(
                set(candidate) == {"recipe_sequence"}, "candidate_field_boundary", f"{key}/{side}"
            )
            check(
                all(set(r) <= RECIPE_KEYS for r in candidate["recipe_sequence"]),
                "canonical_recipe_keys",
                f"{key}/{side}",
            )
            check(
                not FORBIDDEN_KEYS.intersection(all_keys(candidate)),
                "candidate_no_scores_or_identity_keys",
                f"{key}/{side}",
            )
            check(
                not re.search(r"r0-\d+|exp-\d+|/home/|ckpts/|runs/", json.dumps(candidate)),
                "candidate_no_identity_or_paths",
                f"{key}/{side}",
            )
        check(
            expand_catalog(judge.compact_payload(payload)) == payload,
            "lossless_catalog_roundtrip",
            f"{arm}/{key}",
        )
        path = judge_dir / "outputs" / arm / f"{key}.json"
        if not path.exists():
            continue
        try:
            result = read(path)
        except json.JSONDecodeError:
            incomplete_outputs.append(str(path))
            continue
        outputs[f"{arm}_present"] += 1
        check(
            all(result.get(k) == job[k] for k in ("id", "arm", "input_sha256", "swapped")),
            "output_matches_frozen_job",
            f"{arm}/{key}",
        )
        models.add(result.get("model_requested"))
        if result.get("valid"):
            outputs[f"{arm}_valid"] += 1
            try:
                decoded = judge.decode_prediction(result["raw_response"], swap)
                check(
                    all(result.get(k) == decoded[k] for k in ("p_a", "choice_a", "rationale")),
                    "output_orientation_and_decode",
                    f"{arm}/{key}",
                )
            except (ValueError, KeyError, TypeError):
                check(False, "output_orientation_and_decode", f"{arm}/{key}")
        else:
            outputs[f"{arm}_invalid"] += 1
    check(len(models) <= 1 and None not in models, "consistent_requested_judge_model")

    swap_protocol_path = judge_dir / "swap_protocol.json"
    if swap_protocol_path.exists():
        swap_protocol = read(swap_protocol_path)
        selected = sorted(primary, key=lambda key: judge.digest(["swap-audit", key]))[:10]
        check(swap_protocol["pair_ids"] == selected, "swap_selection_independent_of_outcomes")
        check(
            {(j["id"], j["arm"]) for j in all_jobs if j["arm"].endswith("_swap")}
            == {(key, arm + "_swap") for key in selected for arm in ("within_run", "cross_run")},
            "swap_audit_job_coverage",
        )

    feasibility = {}
    for gap in (0, 0.01, 0.02):
        pairs = [(a, b) for a, b, delta in all_siblings if delta >= gap]
        feasibility[str(gap)] = {
            "pairs": len(pairs),
            "cells": len({a["cell_id"] for a, _ in pairs}),
            "candidate_cards": len({r["example_id"] for pair in pairs for r in pair}),
        }
    candidates = {r["example_id"] for a, b, _ in all_siblings for r in (a, b)}
    snapshot_counts = Counter()
    for eid in candidates:
        row = by_id[eid]
        manifest = read((args.raw_root / row["source_card"]).parent / "snapshot/MANIFEST.json")
        files = manifest["files"]
        snapshot_counts["candidate_cards"] += 1
        snapshot_counts["cards_with_snapshot_files"] += bool(files)
        snapshot_counts["cards_with_any_file_timestamp_at_or_before_plan"] += any(
            f["at"] <= row["first_submitted_at"] for f in files
        )
    check(snapshot_counts["candidate_cards"] == 85, "sibling_snapshot_cohort")
    check(
        snapshot_counts["cards_with_any_file_timestamp_at_or_before_plan"] == 0,
        "no_verified_decision_time_script_snapshot",
    )
    first_measurement = {}
    for eid in candidates:
        times = [
            record["at"]
            for record in records[eid]
            if any(
                isinstance(m, dict)
                and m.get("value") is not None
                and any(
                    word in (str(m.get("name") or "") + " " + str(m.get("metric") or "")).lower()
                    for word in ("accuracy", "gsm8k")
                )
                for m in (record["card"].get("result") or {}).get("measurements") or []
            )
        ]
        first_measurement[eid] = min(times) if times else None
    superficially_contemporaneous = set()
    for a, b, _ in all_siblings:
        earlier, later = sorted((a, b), key=lambda r: r["first_submitted_at"])
        measured = [
            first_measurement[r["example_id"]] for r in (a, b) if first_measurement[r["example_id"]]
        ]
        if measured and later["first_submitted_at"] < min(measured):
            superficially_contemporaneous.add((earlier["example_id"], later["example_id"]))
    check(
        superficially_contemporaneous == {(a, b) for a, b, _, _ in FALSE_CONTEMPORANEOUS},
        "only_five_delayed_recording_cases",
    )
    false_cases = []
    for earlier, later, launched, marker in FALSE_CONTEMPORANEOUS:
        first = records[later][0]
        statement = first["card"]["problem"]["statement"]
        check(marker in statement, "later_plan_records_earlier_result", later)
        check(
            launched < by_id[later]["first_submitted_at"],
            "audited_launch_precedes_later_plan",
            later,
        )
        false_cases.append(
            {
                "pair_id": pair_id(earlier, later),
                "earlier": earlier,
                "later": later,
                "in_primary_cohort": pair_id(earlier, later) in primary,
                "later_plan_source": str(
                    (args.raw_root / by_id[later]["source_card"]).parent / "record-01.json"
                ),
                "later_plan_earlier_result_marker": marker,
                "earlier_launch_at": launched,
                "launch_evidence": "Previously audited assistant launch command in solve_out_sanitized.txt; this verifier does not reparse traces.",
            }
        )
    source_paths = [
        args.examples,
        *sorted(Path(__file__).parent.glob("rpm_*.py")),
        Path(__file__).parent / "build_examples.py",
        Path(__file__).parent / "benchmark.py",
    ]
    protocol_paths = [
        judge_dir / "protocol.json",
        judge_dir / "folds.json",
        judge_dir / "jobs.json",
        *(root / name / "fold_assignments.json" for name in LEARNED_DIRS),
    ]
    if swap_protocol_path.exists():
        protocol_paths.append(swap_protocol_path)
    return {
        "schema": "rpm-v2-verification-v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_passed": not errors,
        "errors": errors,
        "checks": dict(checks),
        "counts": {
            "eligible_checkpoints": len(rows),
            "primary_pairs": len(primary),
            "gap_2pp_pairs": sum(abs(a["y"] - b["y"]) >= 0.02 for a, b in primary.values()),
            "primary_cells": len({a["cell_id"] for a, _ in primary.values()}),
            "pairs_with_history": sum(bool(h) for h in history.values()),
            "history_histogram": dict(history_counts),
            "jobs": len(jobs),
            "auxiliary_jobs": len(all_jobs) - len(jobs),
        },
        "learned_coverage": learned_coverage,
        "judge_output_coverage": dict(outputs),
        "all_judge_outputs_present": sum(
            outputs[f"{arm}_present"] for arm in ("within_run", "cross_run")
        )
        == 178,
        "all_judge_outputs_valid": sum(
            outputs[f"{arm}_valid"] for arm in ("within_run", "cross_run")
        )
        == 178,
        "incomplete_output_files_at_read_time": incomplete_outputs,
        "judge_models_requested": sorted(m for m in models if m is not None),
        "sibling_feasibility": {
            "retrospective_nonmerge_by_gap": feasibility,
            "script_snapshots": dict(snapshot_counts),
            "verified_simultaneous_choice_batches": 0,
            "delayed_recording_false_contemporaneous_cases": false_cases,
            "qualification": "Prior audit found all remaining pairs already had an earlier recorded outcome before the later proposal. Final script snapshots do not establish decision-time code.",
        },
        "source_file_sha256": {str(p): file_hash(p) for p in source_paths},
        "protocol_file_sha256": {str(p): file_hash(p) for p in protocol_paths},
        "limitations": [
            "Hashes describe current checked files, not a claim they were frozen before every earlier analysis.",
            "Available outputs are checked for provenance/decoding only; predictive performance is not computed.",
            "Missing or concurrently incomplete output files are coverage gaps, not successful completed judgments.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl")
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/traj/raw/awm-gsm8k-trajectories")
    )
    parser.add_argument("--rpm-root", type=Path, default=Path("data/analysis/rpm"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis/rpm/verification.json"))
    args = parser.parse_args()
    report = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "verification_passed",
                    "counts",
                    "judge_output_coverage",
                    "all_judge_outputs_present",
                    "errors",
                )
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["verification_passed"] else 1)


if __name__ == "__main__":
    main()
