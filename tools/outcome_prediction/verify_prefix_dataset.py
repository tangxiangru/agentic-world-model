"""Verify extraction invariants without treating unresolved inputs as complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tools.outcome_prediction.prefix_dataset import (
    administrative_card_source,
    digest,
    jsonl,
    read_json,
)
from tools.outcome_prediction.prefix_labels import load_label
from tools.outcome_prediction.rpm_code_provenance import timestamp


def verify(dataset):
    dataset = Path(dataset)
    manifest = read_json(dataset / "manifest.json")
    mirror = Path(manifest["mirror"])
    sources = {
        item["id"]: item
        for item in read_json(mirror / "rescore10/relay/gcs_manifest.json")["checkpoints"]
    }
    groups = {
        name: {item["exp_id"]: item for item in jsonl(dataset / f"{name}.jsonl")}
        for name in ("inputs", "labels", "audit", "examples")
    }
    errors = []
    counts = Counter()

    def require(condition, message):
        if not condition:
            errors.append(message)

    ids = set(groups["inputs"])
    for name, rows in groups.items():
        require(set(rows) == ids, f"{name}: record IDs disagree")
        require(len(jsonl(dataset / f"{name}.jsonl")) == len(rows), f"{name}: duplicate IDs")
    require(ids <= set(sources), "output contains unknown checkpoint IDs")
    require(manifest["counts"]["inputs"] == len(ids), "input count inconsistent")
    require(
        manifest.get("source_code_unchanged_during_build") is True,
        "source code changed during extraction",
    )
    for filename, expected in manifest["source_sha256"].items():
        path = Path(__file__).with_name(filename)
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            f"extractor code has changed: {filename}",
        )
    for exp_id in sorted(ids):
        row, audit, label, example = (
            groups[name][exp_id] for name in ("inputs", "audit", "labels", "examples")
        )
        payload = row["input"]
        require(
            set(payload)
            == {"benchmark", "base_model", "prefix_recipes", "generation_config", "evaluation"},
            f"{exp_id}: unexpected input keys",
        )
        require(
            digest(payload) == row["input_sha256"] == audit["input_sha256"],
            f"{exp_id}: changed input hash",
        )
        require(
            example["input"] == payload and example["output"] == label["output"],
            f"{exp_id}: joined example disagrees",
        )
        require(
            len(payload["prefix_recipes"]) == len(audit["recipes"]),
            f"{exp_id}: recipe audit length",
        )
        cutoff = timestamp(audit["archive_cutoff"])
        last = None
        for recipe, evidence in zip(payload["prefix_recipes"], audit["recipes"]):
            require(set(recipe) == {"exp_id", "codes"}, f"{exp_id}: non-code recipe metadata")
            require(recipe["exp_id"] == evidence["exp_id"], f"{exp_id}: recipe identity mismatch")
            require(
                len(recipe["codes"]) == len(evidence["code_evidence"]),
                f"{exp_id}: code audit length",
            )
            for code, event in zip(recipe["codes"], evidence["code_evidence"]):
                require(
                    set(code) == {"kind", "path", "content"} and isinstance(code["content"], str),
                    f"{exp_id}: malformed code",
                )
                require(
                    code["kind"] in {"file_version", "shell_command"},
                    f"{exp_id}: non-code tool result",
                )
                require(
                    code["kind"] != "file_version"
                    or not administrative_card_source(code["content"]),
                    f"{exp_id}: administrative card helper in prediction input",
                )
                available = timestamp(event["available_at"])
                require(available < cutoff, f"{exp_id}: code at/after archive cutoff")
                if event.get("snapshot_path"):
                    snapshot = Path(event["snapshot_path"])
                    require(
                        hashlib.sha256(snapshot.read_bytes()).hexdigest()
                        == event["source_content_sha256"],
                        f"{exp_id}: source snapshot changed",
                    )
                require(last is None or last <= available, f"{exp_id}: unordered code prefix")
                last = available
                counts["code_entries_checked"] += 1
        actual_label = load_label(
            mirror / "rescore10/results" / f"{exp_id}.json", payload["benchmark"]
        )
        require(
            actual_label == label["output"], f"{exp_id}: label does not match source recomputation"
        )
        if actual_label["status"] == "complete":
            rates = actual_label["per_run_pass_rate"]
            require(
                len(rates) == 10
                and math.isclose(sum(rates) / 10, actual_label["avg_pass_rate"], abs_tol=1e-12),
                f"{exp_id}: not ten-run mean",
            )
        counts["labels_" + actual_label["status"]] += 1
        status = row["quality"]["generation_config_status"]
        require(
            payload["generation_config"] == audit["generation_config"]["generation_config"],
            f"{exp_id}: config audit mismatch",
        )
        if status == "unknown":
            require(payload["generation_config"] is None, f"{exp_id}: guessed unknown config")
        else:
            require(
                isinstance(payload["generation_config"], dict), f"{exp_id}: missing known config"
            )
        counts["generation_" + status] += 1
        counts["rows"] += 1
    inventory = read_json(dataset / "trajectories.json")
    require(
        set().union(*(set(item["exp_ids"]) for item in inventory)) == ids,
        "trajectory grouping does not cover all output rows",
    )
    counts["trajectories_in_inventory"] = len(inventory)
    unscored = jsonl(dataset / "unscored_trajectories.jsonl")
    require(
        {item["trajectory_id"] for item in unscored}
        == {item["trajectory_id"] for item in inventory if not item["exp_ids"]},
        "unscored trajectory extraction coverage mismatch",
    )
    for item in unscored:
        require(
            item["output"]["avg_pass_rate"] is None and item["output"]["n_runs"] == 0,
            "unscored trajectory has fabricated ten-run label",
        )
        for recipe in item["input"]["prefix_recipes"]:
            for code in recipe["codes"]:
                require(
                    code["kind"] != "file_version"
                    or not administrative_card_source(code["content"]),
                    "unscored trajectory contains an administrative card helper",
                )
    counts["unscored_trajectory_prefixes"] = len(unscored)
    return {
        "extraction_invariants_passed": not errors,
        "errors": errors,
        "counts": dict(counts),
        "requested_dataset_complete": False,
        "not_proven_by_this_verifier": [
            "Semantic outcome-free content review",
            "Complete executed recipe and checkpoint binding",
            "Missing generation configs",
            "Actual archived configs and effective serving state",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = verify(args.dataset)
    if args.write_report:
        (args.dataset / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["extraction_invariants_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
