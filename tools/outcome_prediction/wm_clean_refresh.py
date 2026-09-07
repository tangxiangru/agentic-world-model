"""Prepare a pinned, labeled-only, conservatively screened recorder dataset.

This is a DATA refresh, not another benchmark or a full artifact certification.
Raw inventory, official labels, and predictor inputs are separate. Neither train
nor test loaders accept unlabeled/quarantined targets. Existing studies stay frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tools.outcome_prediction.wm_code_benchmark import positive_input
from tools.outcome_prediction.wm_dataset import extract_recorder_dataset, stratified_run_split
from tools.outcome_prediction.wm_grade_inventory import valid_official_label

REPOSITORY = "JerrrrryL/awm-gsm8k-trajectories"
REVISION = "01406da734fb9016530bcdfeee027e3760587c6e"
RAW_ROOT = Path("data/traj/raw/awm-gsm8k-trajectories-01406da734fb")
OUTPUT = Path("data/analysis/wm_clean/01406da734fb_v2")
PREVIOUS = Path("data/analysis/wm_rpm/data_v3/private/inventory.jsonl")
SOURCES = (
    ("manifest_r0.json", "gsm8k", "google/gemma-3-4b-pt", 1319),
    ("manifest_aime_r0.json", "aime2025", "Qwen/Qwen3-4B-Base", 30),
    ("manifest_aime2_r0.json", "aime2025", "Qwen/Qwen3-4B-Base", 30),
)
FINGERPRINT_FIELDS = (
    "first_record_sha256",
    "card_sha256",
    "official_metric_sha256",
    "ledger_sha256",
)
SEED = 20260905
# Existing source reviews found conditional shipment/selection despite matching
# setup and output strings. A changed revision does not automatically clear a hold.
REVIEW_HOLDS = {
    "aime-r0-20/exp-07": "eea358e0d47a747f8f716bee2d7d947cba5fbe77641560eb6466eab3bba48f28",
    "aime-r0-20/exp-08": "eb88ee1b8c81db8cc1872e0fd74a70231596f6730a08e90db1a3f21fdbf35259",
    "aime-r0-20/exp-09": "d74ea1c7028ea931f97719dfb2bf883f1eb7256ba067b397e1aae0a6c80def45",
    "aime-r0-20/exp-11": "b870116aa2cb1c0d117782ecd130a9b7607595eec2e157b7c1fad56ea5987581",
    "aime-r0-26/exp-05": "29bd5bd714573b9bce7d5a535035f41ca7d7cfc0edff163bea82063a2fd3e0cd",
    "aime-r0-26/exp-06": "091a2150d8cfde153d0aad0af3069b0987d8f785c588506a27ae71d08c61536f",
}
REVIEW_FILES = {
    "manual_source_review_train_aime20_v1.json": "eaf64f088161c9e0a95286bee3d52c20b59f6e7e5e9ffa7b941b3984ffe7d926",
    "whole_step_review_train_aime20_v1.json": "d339c0bb7db9fa7189747866572d4c9d8dca59664df22a12b67c9fb4831ba4bd",
    "manual_source_review_test_aime26_v1.json": "88ee0d5968efb49e5f04ef082422d362539764afb3a88e17f37c60a181d26243",
    "whole_step_review_test_aime26_v1.json": "daac8378209107690395be80a41a2840bbd718a071367e9aebeef7c0d19df1fd",
    "plan_narrative_review_v1.json": "4a346e48d79aade83951af01a4f6c2afe7cbdb06c0916f7deed12332edfa2ff7",
    "plan_narrative_ship_review_v1_hold.json": "cc33ab66be118cadf1caa607a961098d9ec5828d9acfec8874bade1da350978d",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _json_finite(value):
    """Retain malformed nonfinite values as audit strings, never numeric targets.

    Original bytes remain in the raw download. This only makes the audit copy
    standard JSON so one corrupt unlabelled row cannot block all clean records.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_number": repr(value)}
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_finite(v) for v in value]
    return value


def _same_source(row, previous):
    current, old = row.get("provenance", {}), previous.get("provenance", {})

    def evidence(item):
        return [
            {k: v for k, v in entry.items() if k != "trace_path"}
            for entry in item.get("audit", {}).get("code_provenance", [])
        ]

    return all(current.get(k) and current[k] == old.get(k) for k in FINGERPRINT_FIELDS) and (
        evidence(row) == evidence(previous)
    )


def screen_row(row, previous=None):
    """No valid score magnitude, prediction, or local outcome affects selection.

    Unknown path aliases and even benign setup edits are quarantined for review,
    not called proven corruption. Missing code is unknown, not model failure.
    """
    reasons = []
    audit = row.get("audit")
    if not isinstance(audit, dict):
        audit = {}
        reasons.append("missing_or_invalid_audit")
    if not valid_official_label(row.get("label")):
        reasons.append("missing_or_invalid_official_label")
    if row.get("eligible") is not True or audit.get("eligible") is not True:
        reasons.append("failed_structural_screen")
    reasons.extend("structural:" + str(r) for r in audit.get("reasons", []))
    if row.get("first_stage") != "plan":
        reasons.append("first_submission_not_plan")
    try:
        at = datetime.fromisoformat(row["first_submitted_at"].replace("Z", "+00:00"))
        if at.tzinfo is None:
            raise ValueError("naive timestamp")
    except (AttributeError, KeyError, TypeError, ValueError):
        reasons.append("invalid_proposal_timestamp")
    if audit.get("first_missing_fields"):
        reasons.append("incomplete_first_registration")
    flags = audit.get("label_fidelity_flags")
    if not isinstance(flags, list):
        reasons.append("missing_fidelity_audit")
    else:
        reasons.extend("unresolved_fidelity:" + str(flag) for flag in flags)
    if audit.get("first_final_setup_changed") is not False:
        reasons.append("setup_equality_not_established")
    if audit.get("first_final_setup_changed_fields"):
        reasons.append("setup_change_fields_recorded")
    if audit.get("output_artifact_path_mismatch") is not False:
        reasons.append("output_equality_not_established")
    binding = audit.get("output_artifact_comparison") or {}
    if not isinstance(binding, dict):
        binding = {}
    planned, produced = (
        binding.get("first_declared_output_dir"),
        binding.get("final_output_checkpoint"),
    )
    if not (
        isinstance(planned, str)
        and planned.strip()
        and isinstance(produced, str)
        and produced.strip()
        and planned.rstrip("/") == produced.rstrip("/")
    ):
        reasons.append("unresolved_output_binding")
    if row.get("example_id") in REVIEW_HOLDS:
        reasons.append("unresolved_prior_target_binding_hold")
    status = "no_previous_record"
    if previous is not None:
        if previous.get("example_id") != row.get("example_id"):
            raise ValueError("Previous record identity mismatch")
        matched = _same_source(row, previous)
        status = "source_matched" if matched else "source_changed_or_unbound"
        if previous.get("audit", {}).get("eligible") is False or previous.get("eligible") is False:
            reasons.append("previous_rejection" if matched else "prior_rejection_requires_review")
    return {
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "previous_review_status": status,
    }


def project_example(row):
    """Recipe material for fixed positive feature extractors, not raw LLM prose.

    No labels, generic ancestor results, final snapshots, or audit status. These
    recipe strings/code still require the existing numeric feature whitelist;
    they are not certified safe for an unrestricted text encoder/LLM.
    """
    result = {k: row[k] for k in ("example_id", "cell_id", "benchmark", "scientist_model")}
    result["model_input"] = positive_input(row)
    return result


def freeze_dataset(rows, output, previous_rows=(), source_metadata=None, test_count=4):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new versioned dataset directory")
    rows = list(rows)
    by_id = {r["example_id"]: r for r in rows}
    previous_rows = list(previous_rows)
    previous = {r["example_id"]: r for r in previous_rows}
    if len(by_id) != len(rows) or len(previous) != len(previous_rows):
        raise ValueError("Duplicate example identity")
    # Split ALL sessions before filtering, so selection does not change membership.
    split = stratified_run_split(rows, test_count=test_count, seed=SEED)
    decisions = {r["example_id"]: screen_row(r, previous.get(r["example_id"])) for r in rows}
    selected = [r for r in rows if decisions[r["example_id"]]["eligible"]]
    examples = [project_example(r) for r in selected]
    labels = {r["example_id"]: copy.deepcopy(r["label"]) for r in selected}
    split["clean_train_example_ids"] = [
        r["example_id"] for r in selected if split["cell_partition"][r["cell_id"]] == "train"
    ]
    split["clean_test_example_ids"] = [
        r["example_id"] for r in selected if split["cell_partition"][r["cell_id"]] == "test"
    ]
    arms = sorted({(r.get("provenance") or {}).get("manifest_name", r["benchmark"]) for r in rows})
    summary = {}
    for arm in arms:
        part = [
            r
            for r in rows
            if (r.get("provenance") or {}).get("manifest_name", r["benchmark"]) == arm
        ]
        kept = [r for r in part if decisions[r["example_id"]]["eligible"]]
        summary[arm] = {
            "entries": len(part),
            "sessions": len({r["cell_id"] for r in part}),
            "valid_official_labels": sum(valid_official_label(r.get("label")) for r in part),
            "clean_labeled": len(kept),
            "clean_train": sum(split["cell_partition"][r["cell_id"]] == "train" for r in kept),
            "clean_test": sum(split["cell_partition"][r["cell_id"]] == "test" for r in kept),
            "quarantined_labeled": sum(
                valid_official_label(r.get("label")) and not decisions[r["example_id"]]["eligible"]
                for r in part
            ),
            "unlabeled_or_invalid": sum(not valid_official_label(r.get("label")) for r in part),
            "clean_with_reconstructed_main_code": sum(
                any(
                    c.get("role") == "training" and c.get("status") == "reconstructed"
                    for c in r["model_input"].get("code", [])
                )
                for r in kept
            ),
        }
    sources = [
        Path(__file__),
        *[
            Path("tools/outcome_prediction") / name
            for name in (
                "wm_dataset.py",
                "rpm_code_provenance.py",
                "build_examples.py",
                "wm_grade_inventory.py",
                "wm_code_benchmark.py",
            )
        ],
    ]
    policy = {
        "schema": "wm-clean-labeled-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "data_preparation_only_no_new_fits_or_predictions",
        "inclusion": "Valid official label, complete prospective registration, structural pass, unchanged setup and literal known output match, no unresolved prior rejection. No score-magnitude filtering.",
        "quarantine": "Conservative exclusions include potentially benign setup edits or aliases. They are not all proven dirty; adjudication can recover them in a future version.",
        "code": "Only successful trace-reconstructed pre-proposal code. Missing code stays unknown; mutable final snapshots are never substituted.",
        "features": "Use fixed positive numeric extractors. Exported setup/code is not a license to embed unrestricted free text.",
        "ancestry": "This bundle exports current recipes only, not a certified full-history or delta-model dataset. Add only reviewed same-session earlier ancestors; exact consumed-checkpoint identity must be checked before joining a parent score. Never invent base accuracy or import generic ancestor scores.",
        "split": "Whole sessions; benchmark/scientist-stratified; identities only; assigned before filtering.",
        "baseline": "Old cached LLM forecasts lack new AIME2 rows and have old training banks/splits. They are not a comparable refreshed baseline.",
        "limits": "Automated strict consistency screen, not end-to-end recipe execution or checkpoint-byte certification. Release summaries and old results were previously visible. No fresh-test claim.",
        "raw_preserved": True,
        "prior_target_binding_holds": REVIEW_HOLDS,
        "source_metadata": source_metadata,
        "source_code": {str(p.resolve()): sha(p) for p in sources},
    }
    output.mkdir(parents=True, mode=0o700)
    (output / "private").mkdir(mode=0o700)
    with (output / "private/inventory.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(_json_finite(row), sort_keys=True, allow_nan=False) + "\n")
    (output / "private/inventory.jsonl").chmod(0o600)
    for name, value in (
        ("clean_inputs.json", examples),
        ("clean_labels.json", labels),
        ("decisions.json", decisions),
        ("split.json", split),
        ("policy.json", policy),
        ("summary.json", summary),
        (
            "exclusion_counts.json",
            dict(Counter(reason for d in decisions.values() for reason in d["reasons"])),
        ),
    ):
        write(output / name, value)
    write(
        output / "manifest.json",
        {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    return summary


def load_partition(bundle, partition):
    """The shared train/eval gate: verify frozen data and reject missing labels."""
    if partition not in {"train", "test"}:
        raise ValueError("Partition must be train or test")
    bundle = Path(bundle)
    manifest = read(bundle / "manifest.json")
    required = {
        "clean_inputs.json",
        "clean_labels.json",
        "decisions.json",
        "split.json",
        "policy.json",
    }
    if not required <= manifest.keys():
        raise ValueError("Incomplete dataset manifest")
    for name, expected in manifest.items():
        path = bundle / name
        if not path.resolve().is_relative_to(bundle.resolve()) or sha(path) != expected:
            raise ValueError("Modified frozen dataset: " + name)
    inputs, labels = read(bundle / "clean_inputs.json"), read(bundle / "clean_labels.json")
    decisions, split = read(bundle / "decisions.json"), read(bundle / "split.json")
    ids = [r["example_id"] for r in inputs]
    if len(ids) != len(set(ids)) or set(ids) != set(labels):
        raise ValueError("Input/label identity mismatch")
    if set(split["train_cell_ids"]) & set(split["test_cell_ids"]):
        raise ValueError("Train/test session overlap")
    expected_ids = set(split["clean_" + partition + "_example_ids"])
    selected = []
    for row in inputs:
        key, cell = row["example_id"], row["cell_id"]
        if not key.startswith(cell + "/") or decisions.get(key, {}).get("eligible") is not True:
            raise ValueError("Unclean or mismatched example")
        if not valid_official_label(labels[key]):
            raise ValueError("Missing/invalid supervised label")
        row_partition = split["cell_partition"].get(cell)
        if row_partition not in {"train", "test"} or cell not in split[row_partition + "_cell_ids"]:
            raise ValueError("Unknown or conflicting session partition")
        if row_partition == partition:
            selected.append(row)
    if {r["example_id"] for r in selected} != expected_ids:
        raise ValueError("Clean partition identity mismatch")
    return selected, {r["example_id"]: labels[r["example_id"]]["accuracy"] for r in selected}


def verify_download(raw_root, revision):
    """Check recorder files against pinned HF Git blobs/LFS SHA, not README counts."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(REPOSITORY, revision=revision, files_metadata=True)
    manifests = {source[0] for source in SOURCES}
    cells = {r["cell_id"] for name in manifests for r in read(raw_root / name)}
    files = []
    for entry in info.siblings:
        parts = entry.rfilename.split("/")
        if entry.rfilename not in manifests | {"README.md"} and not (
            len(parts) >= 3 and parts[0] == "cells" and parts[1] in cells
        ):
            continue
        path = raw_root / entry.rfilename
        raw = path.read_bytes()
        if len(raw) != entry.size:
            raise ValueError("Downloaded file size mismatch: " + entry.rfilename)
        if entry.lfs is not None:
            actual, expected = hashlib.sha256(raw).hexdigest(), entry.lfs.sha256
        else:
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            expected = entry.blob_id
        if actual != expected:
            raise ValueError("Downloaded file hash mismatch: " + entry.rfilename)
        files.append(
            {"path": entry.rfilename, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    return {
        "repository": REPOSITORY,
        "revision": info.sha,
        "files": files,
        "total_bytes": sum(f["size"] for f in files),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--previous-inventory", type=Path, default=PREVIOUS)
    args = parser.parse_args()
    metadata = verify_download(args.raw_root, args.revision)
    metadata["prior_target_binding_review_files"] = {}
    for name, expected in REVIEW_FILES.items():
        path = Path("data/analysis/wm_rpm") / name
        if sha(path) != expected:
            raise ValueError("Changed prior source review: " + name)
        metadata["prior_target_binding_review_files"][str(path)] = expected
    metadata["previous_inventory"] = {
        "path": str(args.previous_inventory),
        "sha256": sha(args.previous_inventory),
    }
    rows, audits = [], []
    for manifest, benchmark, base, n in SOURCES:
        part, audit = extract_recorder_dataset(
            raw_root=args.raw_root,
            manifest_name=manifest,
            source_revision=args.revision,
            benchmark=benchmark,
            base_model=base,
            evaluation_n=n,
        )
        rows.extend(part)
        audits.append(audit)
    metadata["source_audits"] = audits
    summary = freeze_dataset(rows, args.output, load_jsonl(args.previous_inventory), metadata)
    for partition in ("train", "test"):
        load_partition(args.output, partition)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
