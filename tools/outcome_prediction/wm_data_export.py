"""Export current reviewed TRAIN supervision and outcome-free TEST inputs.

This is a coverage-preserving dataset snapshot, not final study/cohort approval.
Only TRAIN inventory records are decoded. TEST labels never enter these files.
Case readiness uses input coverage only, never checkpoint score or label values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.outcome_prediction import wm_final_prepare as prepare
from tools.outcome_prediction.wm_final_model import approved_payload
from tools.outcome_prediction.wm_parameter_pilot import load_train_inventory

WRAPPER_KEYS = ("example_id", "cell_id", "model_input", "content_review", "step_sources")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def build(bundle, inventory, split_path, output_dir):
    bundle, inventory, split_path, output = map(Path, (bundle, inventory, split_path, output_dir))
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a NEW immutable data snapshot directory")
    provenance_raw = (bundle / "provenance.json").read_bytes()
    proof = json.loads(provenance_raw)
    for name, expected in proof["files_sha256"].items():
        if name not in {"drafts.json", "review_queue.json", "audit.json"}:
            raise ValueError("Unexpected source artifact file")
        if _sha((bundle / name).read_bytes()) != expected:
            raise ValueError("Stale source artifact hash")
    if set(proof["files_sha256"]) != {"drafts.json", "review_queue.json", "audit.json"}:
        raise ValueError("Incomplete source artifact manifest")
    split_raw = split_path.read_bytes()
    if _sha(split_raw) != proof["split_file_sha256"]:
        raise ValueError("Frozen split hash mismatch")
    if _sha(inventory.read_bytes()) != proof["inventory_sha256"]:
        raise ValueError("Inventory hash mismatch")
    split = json.loads(split_raw)
    train_cells, test_cells = prepare.split_cells(split)
    if split.get("cell_partition") != {
        **dict.fromkeys(train_cells, "train"),
        **dict.fromkeys(test_cells, "test"),
    }:
        raise ValueError("Identity loader partition differs from frozen split")
    drafts = json.loads((bundle / "drafts.json").read_text())
    if len({d["example_id"] for d in drafts}) != len(drafts):
        raise ValueError("Duplicate source recipe")
    for draft in drafts:
        expected = "train" if draft["cell_id"] in train_cells else "test"
        if draft["cell_id"] not in train_cells | test_cells or draft["partition"] != expected:
            raise ValueError("Source draft differs from frozen partition")
    training_inventory, skipped = load_train_inventory(inventory, split)
    records, label_audit = prepare.join_train_labels(
        {"drafts": drafts, "audit": json.loads((bundle / "audit.json").read_text())},
        training_inventory,
        split,
    )
    test_inputs = {}
    for draft in drafts:
        if draft["partition"] != "test" or draft["status"] != "approved":
            continue
        approved_payload(draft)
        wrapper = {key: draft[key] for key in WRAPPER_KEYS}
        test_inputs[draft["example_id"]] = wrapper
    cases = []
    for cell in sorted(test_cells):
        ids = sorted(
            (key for key, row in test_inputs.items() if row["cell_id"] == cell),
            key=lambda key: _sha(f"20260905:{key}".encode()),
        )
        source = [d for d in drafts if d["cell_id"] == cell]
        cases.append(
            {
                "case_id": cell,
                "cell_id": cell,
                "candidate_ids": ids,
                "input_comparison_ready": len(ids) >= 2,
                "inventory_targets": len(source),
                "pending_target_ids": sorted(
                    d["example_id"] for d in source if d["status"] != "approved"
                ),
                "heldout_label_availability_checked": False,
            }
        )
    coverage = {
        "inventory_cards": len(drafts),
        "train_recipes_with_final_official_labels": len(records),
        "test_input_recipes": len(test_inputs),
        "current_input_comparison_runs": sum(c["input_comparison_ready"] for c in cases),
        "frozen_train_runs": len(train_cells),
        "frozen_test_runs": len(test_cells),
        "train_label_audit": label_audit,
        "test_inventory_rows_skipped_before_record_decode": sum(skipped.values()),
        "run_coverage": [
            {
                "cell_id": cell,
                "partition": "train" if cell in train_cells else "test",
                "inventory_targets": sum(d["cell_id"] == cell for d in drafts),
                "approved_inputs": sum(
                    d["cell_id"] == cell and d["status"] == "approved" for d in drafts
                ),
                "supervised_train_targets": Counter(r["cell_id"] for r in records)[cell],
            }
            for cell in sorted(train_cells | test_cells)
        ],
    }
    encoded = {
        "train.jsonl": "".join(
            json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for r in records
        ).encode(),
        "test_inputs.json": _json(test_inputs),
        "case_index.json": _json(
            {"cohort_finalized": False, "study_frozen": False, "cases": cases}
        ),
        "coverage.json": _json(coverage),
        "split.json": _json({**split, "empty_train_cell_ids": label_audit["empty_train_cell_ids"]}),
    }
    manifest = {
        "schema": "wm-reviewed-data-snapshot-v1",
        "status": "preliminary_coverage_snapshot_not_final_study",
        "source_bundle": str(bundle),
        "source_bundle_provenance_sha256": _sha(provenance_raw),
        "inventory_sha256": proof["inventory_sha256"],
        "frozen_split_sha256": _sha(split_raw),
        "exporter_sha256": _sha(Path(__file__).read_bytes()),
        "test_outcomes_used": False,
        "files_sha256": {name: _sha(raw) for name, raw in encoded.items()},
        "limits": [
            "All original runs remain represented; current review gaps are not permanent cohort exclusions.",
            "TRAIN contains final labels; do not mount it as candidate evidence.",
            "TEST inputs contain no labels or private review audit text.",
            "Comparison readiness is input-only, not packet approval, label availability or executable-byte certification.",
            "Freeze a complete study protocol and exposure-reviewed packets separately before inference/scoring.",
        ],
    }
    output.mkdir(parents=True, mode=0o700)
    for name, raw in {**encoded, "provenance.json": _json(manifest)}.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
        (output / name).chmod(0o600)
    return {
        k: coverage[k]
        for k in (
            "train_recipes_with_final_official_labels",
            "test_input_recipes",
            "current_input_comparison_runs",
            "frozen_train_runs",
            "frozen_test_runs",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "inventory", "split", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.bundle, args.inventory, args.split, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
