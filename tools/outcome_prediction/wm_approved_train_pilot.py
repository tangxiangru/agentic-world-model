"""Private restricted integration pilot for explicitly reviewed TRAIN recipes.

This is not a final WM choice, held-out evaluation, RPM comparison, or proof of
multiway selection quality. Existing content gates and fixed model settings are
unchanged. TEST inventory lines are identified lexically and skipped before
JSON decoding; only approved TRAIN targets have their final labels joined.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy
import scipy
import sklearn

from tools.outcome_prediction import wm_final_model as model_module
from tools.outcome_prediction import wm_final_prepare as prepare
from tools.outcome_prediction import wm_parameter_pilot as parameter_pilot

SCOPE = "restricted_reviewed_cohort_train_only_integration_pilot"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def load_prepared(bundle, inventory_path, split_path):
    """Validate immutable source artifacts without parsing inventory records."""
    bundle = Path(bundle)
    provenance = json.loads((bundle / "provenance.json").read_text())
    for name, expected in provenance["files_sha256"].items():
        if Path(name).name != name or file_hash(bundle / name) != expected:
            raise ValueError("Prepared artifact hash mismatch")
    if file_hash(inventory_path) != provenance["inventory_sha256"]:
        raise ValueError("Prepared inventory hash mismatch")
    if file_hash(split_path) != provenance["split_file_sha256"]:
        raise ValueError("Prepared frozen split hash mismatch")
    if file_hash(prepare.__file__) != provenance["builder_sha256"]:
        raise ValueError("Prepared builder source changed")
    for name, expected in provenance["source_module_sha256"].items():
        if file_hash(importlib.import_module(name).__file__) != expected:
            raise ValueError("Prepared source module changed")
    for item in provenance["whole_step_reviews"]:
        if file_hash(item["path"]) != item["file_sha256"]:
            raise ValueError("Whole-step review artifact changed")
    prepared = {
        "schema": provenance["schema"],
        "drafts": json.loads((bundle / "drafts.json").read_text()),
        "review_queue": json.loads((bundle / "review_queue.json").read_text()),
        "audit": json.loads((bundle / "audit.json").read_text()),
    }
    if model_module.digest(prepared) != provenance["result_sha256"]:
        raise ValueError("Prepared result hash mismatch")
    return prepared, provenance


def cohort_coverage(prepared, train_rows, records, join_audit, split):
    """Report every TRAIN target/run without consulting omitted target labels."""
    train_cells, test_cells = prepare.split_cells(split)
    drafts = {}
    for draft in prepared["drafts"]:
        if draft["cell_id"] in test_cells:
            continue
        if draft["cell_id"] not in train_cells or draft["example_id"] in drafts:
            raise ValueError("Unknown or duplicate draft identity")
        drafts[draft["example_id"]] = draft
    identities = []
    for row in train_rows:
        if row["cell_id"] in test_cells:
            continue
        if row["cell_id"] not in train_cells:
            raise ValueError("Unknown inventory run")
        identities.append((row["example_id"], row["cell_id"]))
    if len(set(identities)) != len(identities) or {i[0] for i in identities} != set(drafts):
        raise ValueError("TRAIN draft/inventory coverage mismatch")
    retained = {r["example_id"] for r in records}
    gaps = {r["example_id"]: r["reason"] for r in join_audit["label_gaps"]}
    omitted, run_counts, benchmarks = [], defaultdict(Counter), defaultdict(set)
    for identity, cell in sorted(identities):
        draft = drafts[identity]
        run_counts[cell]["inventory_targets"] += 1
        run_counts[cell]["approved_targets"] += int(draft["status"] == "approved")
        run_counts[cell]["retained_targets"] += int(identity in retained)
        if identity not in retained:
            omitted.append(
                {
                    "example_id": identity,
                    "cell_id": cell,
                    "draft_status": draft["status"],
                    "reason": gaps.get(identity, "whole_recipe_not_approved"),
                    "draft_issue_kinds": sorted({i["kind"] for i in draft["audit"]["issues"]}),
                }
            )
    for row in records:
        benchmarks[row["model_input"]["task"]["benchmark"]].add(row["cell_id"])
    expected_benchmarks = {d["model_input"]["task"]["benchmark"] for d in drafts.values()}
    return {
        "train_inventory_targets": len(identities),
        "retained_targets": len(records),
        "retained_train_runs": len({r["cell_id"] for r in records}),
        "omitted_targets": omitted,
        "run_coverage": [
            {"cell_id": cell, **dict(run_counts[cell])} for cell in sorted(train_cells)
        ],
        "benchmark_coverage": {
            benchmark: {
                "retained_targets": sum(
                    r["model_input"]["task"]["benchmark"] == benchmark for r in records
                ),
                "retained_runs": len(benchmarks[benchmark]),
                "retained_cell_ids": sorted(benchmarks[benchmark]),
            }
            for benchmark in sorted(expected_benchmarks)
        },
    }


def fit_is_permitted(coverage):
    return bool(coverage["benchmark_coverage"]) and all(
        entry["retained_runs"] >= model_module.SPEC["cv_folds"]
        for entry in coverage["benchmark_coverage"].values()
    )


def _source_hashes():
    paths = [
        Path(__file__),
        Path(model_module.__file__),
        Path(prepare.__file__),
        Path(parameter_pilot.__file__),
        Path(model_module.recipe_input.__file__),
        Path(prepare.compiler.__file__),
        Path(prepare.lineage.__file__),
    ]
    return {str(path.resolve()): file_hash(path) for path in paths}


def build_pilot(bundle, inventory_path, split_path, output_dir):
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a NEW immutable private pilot directory")
    source_hashes = _source_hashes()
    prepared, source_provenance = load_prepared(bundle, inventory_path, split_path)
    split = json.loads(Path(split_path).read_text())
    train_rows, skipped = parameter_pilot.load_train_inventory(inventory_path, split)
    records, join_audit = prepare.join_train_labels(prepared, train_rows, split)
    coverage = cohort_coverage(prepared, train_rows, records, join_audit, split)
    model_split = copy.deepcopy(split)
    model_split["empty_train_cell_ids"] = join_audit["empty_train_cell_ids"]
    model_split["coverage_scope"] = SCOPE
    model_split["frozen_split_sha256"] = file_hash(split_path)
    output.mkdir(parents=True, mode=0o700)
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for record in sorted(records, key=lambda row: row["example_id"]):
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False))
            stream.write("\n")
    (output / "records.jsonl").chmod(0o600)
    _write_json(output / "split.json", model_split)
    _write_json(output / "coverage.json", {**coverage, "label_join": join_audit})
    report = {
        "scope": SCOPE,
        "status": "insufficient_approved_train_runs",
        "final_wm_selected": False,
        "heldout_evaluation_performed": False,
        "rpm_comparison_performed": False,
        "terminal_target_filter_applied": False,
        "fit_guard": "At least four usable TRAIN runs in every represented source benchmark",
        "fixed_model_spec": copy.deepcopy(model_module.SPEC),
        "cohort": {
            key: coverage[key]
            for key in (
                "train_inventory_targets",
                "retained_targets",
                "retained_train_runs",
                "benchmark_coverage",
            )
        },
        "heldout_inventory_lines_skipped_before_json_decode": sum(skipped.values()),
        "limits": [
            "Restricted reviewed cohort, not representative full-dataset evaluation.",
            "Only final target labels are supervised; no ancestor results enter recipe inputs.",
            "Zero selection regret on singleton runs is automatic, not evidence of ranking quality.",
            "Any CV-selected variant is a local integration-pilot choice, not the final WM.",
            "No hyperparameter tuning, held-out scoring, serving, paid calls, or GPU use.",
            "Inventory bytes are read for hashing/identity scan; TEST records are not JSON decoded.",
        ],
    }
    failure = None
    if fit_is_permitted(coverage):
        try:
            fitted = model_module.train(records, model_split)
            model_path = output / "wm.joblib"
            model_hash = fitted.save(model_path)
            model_path.chmod(0o600)
            manifest = {
                **fitted.training_manifest,
                "pilot_scope": SCOPE,
                "final_wm_selected": False,
                "model_sha256": model_hash,
                "train_records_sha256": file_hash(output / "records.jsonl"),
                "split_sha256": file_hash(output / "split.json"),
            }
            _write_json(output / "training_manifest.json", manifest)
            report["status"] = "completed_restricted_train_cv"
            report["benchmark_train_cv"] = {
                benchmark: {
                    key: value[key] for key in ("champion", "candidate_metrics", "selection_scope")
                }
                for benchmark, value in fitted.training_manifest["benchmark_train_cv"].items()
            }
            report["model_sha256"] = model_hash
        except Exception as exc:  # noqa: BLE001 - preserve any fit failure, then re-raise below.
            failure = exc
            report["status"] = "fit_failed_guards_not_weakened"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    if source_hashes != _source_hashes():
        failure = RuntimeError("Source changed during pilot; outputs are not finalized")
        report["status"] = "source_changed_not_finalized"
    _write_json(output / "report.json", report)
    _write_json(
        output / "provenance.json",
        {
            "schema": "wm-approved-train-integration-pilot-v1",
            "scope": SCOPE,
            "visibility": "private_contains_train_final_labels",
            "source_hashes": source_hashes,
            "sources_unchanged": source_hashes == _source_hashes(),
            "source_bundle": str(bundle),
            "source_bundle_provenance_sha256": file_hash(Path(bundle) / "provenance.json"),
            "source_bundle_files_sha256": source_provenance["files_sha256"],
            "inventory_sha256": file_hash(inventory_path),
            "frozen_split_sha256": file_hash(split_path),
            "heldout_record_objects_decoded": 0,
            "heldout_labels_used": False,
            "skipped_heldout_lines_by_cell": skipped,
            "versions": {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "files_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
        },
    )
    if failure is not None:
        raise RuntimeError(
            "Restricted pilot failed; see private report, no guard relaxed"
        ) from failure
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_pilot(args.bundle, args.inventory, args.split, args.output_dir)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
