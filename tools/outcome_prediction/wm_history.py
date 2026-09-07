"""Package disjoint TRAIN-run history for the two inference-only selector arms.

This is historical reference evidence, NEVER a source of WM recipe features.
Candidate runs and every ancestor of a candidate must be outside these runs.
Historical raw traces retain outcomes of those OTHER completed runs. This applies
the user rule per predicted recipe, while retaining the originally requested raw
historical RPM context. It does not approve any candidate input or WM model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.outcome_prediction.wm_corpus import SourceRoot, json_bytes, redact_credentials, sha256
from tools.outcome_prediction.wm_model import digest

GUIDE = """# Historical reference trajectories

These are completed scientist runs from the frozen TRAIN partition, supplied
identically to both selector arms. They are OTHER runs, not steps of a held-out
candidate recipe. Outcomes inside them are historical observations, never known
results of the recipe currently being predicted.

The WM was not trained on these raw transcripts as input features. Its inputs
must instead be separately reviewed, complete recipes with no ancestor outcomes.
Do not treat any historical score as a measurement of a candidate's ancestors.

Each raw_trajectory.txt is the complete archived trace except narrowly redacted
credentials. It contains the scientist's plans, tool calls, code edits and local
observations; the local evaluation protocols can differ from official grading.
Code explicitly recovered before a card's first registration is separately
indexed. Unknown code stays unknown. Historical official labels describe archived
checkpoints of these other runs, not fresh executions or candidate labels.
"""


def validate_history_for_candidates(manifest, *, candidate_run_ids, ancestor_run_ids=()):
    """Refuse history containing any run that contributes to a current candidate."""
    if manifest.get("schema") != "wm-disjoint-raw-history-v1":
        raise ValueError("Unsupported historical corpus")
    candidate_runs = set(candidate_run_ids) | set(ancestor_run_ids)
    if not candidate_runs:
        raise ValueError("Candidate identities are required to validate history")
    if candidate_runs & set(manifest["history_run_ids"]):
        raise ValueError("History would expose outcomes from a candidate recipe's run")
    if not candidate_runs <= set(manifest["intended_candidate_run_ids"]):
        raise ValueError("Candidate is outside the frozen held-out partition")


def package_history(dataset_dir, output_dir):
    """Create an immutable train-only reference corpus; refuse existing output."""
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError("Choose a new history directory")
    with SourceRoot(dataset_dir) as dataset:
        raw_split = dataset.read("split.json")
        split = json.loads(raw_split)
        protocol = json.loads(dataset.read("protocol.json"))
        if digest(split) != protocol["split_sha256"]:
            raise ValueError("Frozen split mismatch")
        train, test = set(split["train_cell_ids"]), set(split["test_cell_ids"])
        if not train or train & test:
            raise ValueError("Invalid whole-run split")
        raw_index = dataset.read("train/raw_trajectory_index.json")
        index = json.loads(raw_index)
        if len(index) != len(train) or {r["cell_id"] for r in index} != train:
            raise ValueError("Raw history must cover exactly the TRAIN runs")
        raw_inventory = dataset.read("private/inventory.jsonl")
        inventory = [json.loads(line) for line in raw_inventory.splitlines() if line.strip()]
        rows = [row for row in inventory if row["cell_id"] in train]
        if {r["cell_id"] for r in rows} != train:
            raise ValueError("Missing training-run inventory")
        if len({r["example_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate historical example ID")
        output = output.parent.resolve(strict=True) / output.name
        source_roots = [Path(s["raw_root"]).resolve(strict=True) for s in protocol["sources"]]
        if any(output.is_relative_to(root) for root in [dataset.path, *source_roots]):
            raise ValueError("History output must be outside source and dataset roots")
        output.mkdir(mode=0o700)
        evidence = output / "evidence"
        evidence.mkdir(mode=0o700)
        files, artifacts = {}, []

        def emit(path, raw, role):
            from tools.outcome_prediction.wm_evidence import _logical_path

            path = _logical_path(path)
            if path in files or redact_credentials(path)[0] != path:
                raise ValueError("Duplicate or credential-bearing output path")
            text, counts = redact_credentials(raw.decode("utf-8"))
            encoded = text.encode("utf-8")
            destination = evidence / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(encoded)
            destination.chmod(0o444)
            files[path] = sha256(encoded)
            artifacts.append(
                {
                    "path": path,
                    "role": role,
                    "source_sha256": sha256(raw),
                    "sha256": files[path],
                    "credential_redactions": counts,
                }
            )

        emit("README.md", GUIDE.encode(), "history_guide")
        sources = {(s["benchmark"], s["source_revision"]): s for s in protocol["sources"]}
        if len(sources) != len(protocol["sources"]):
            raise ValueError("Duplicate historical source")
        for entry in index:
            cell = entry["cell_id"]
            source = sources[(entry["benchmark"], entry["source_revision"])]
            root = Path(source["raw_root"]).absolute()
            expected_path = root / "cells" / cell / "solve_out_sanitized.txt"
            if Path(entry["path"]).absolute() != expected_path:
                raise ValueError("Historical raw index path mismatch")
            with SourceRoot(root) as raw_root:
                raw = raw_root.read(
                    f"cells/{cell}/solve_out_sanitized.txt", expected=entry["sha256"]
                )
                if len(raw) != entry["bytes"]:
                    raise ValueError("Historical trace size mismatch")
                emit(f"runs/{cell}/raw_trajectory.txt", raw, "disjoint_historical_raw_trace")
        card_index = []
        for row in sorted(rows, key=lambda r: r["example_id"]):
            cell, card = row["cell_id"], row["card_id"]
            if row["example_id"] != f"{cell}/{card}":
                raise ValueError("Historical card identity mismatch")
            folder = f"runs/{cell}/cards/{card}"
            recovered = []
            for number, code in enumerate(row["model_input"].get("code", [])):
                item = {key: value for key, value in code.items() if key != "content"}
                item["evidence_path"] = None
                if code.get("content") is not None:
                    if code.get("status") != "reconstructed":
                        raise ValueError("Historical code lacks preproposal reconstruction")
                    name = f"{folder}/recovered_code/{number:03d}.txt"
                    emit(name, code["content"].encode(), "historical_preproposal_code")
                    item["evidence_path"] = name
                recovered.append(item)
            # Deliberately NOT public_payload(row): that legacy function attaches
            # ancestor observations. Historical labels remain separate here.
            emit(
                f"{folder}/first_plan.json",
                json_bytes(
                    {
                        "first_submitted_at": row.get("first_submitted_at"),
                        "first_stage": row.get("first_stage"),
                        "plan": row["model_input"]["plan"],
                        "recovered_code": recovered,
                    }
                ),
                "historical_first_plan_not_candidate_input",
            )
            card_index.append(
                {
                    "example_id": row["example_id"],
                    "cell_id": cell,
                    "benchmark": row["benchmark"],
                    "plan_path": f"{folder}/first_plan.json",
                    "official_accuracy": (row.get("label") or {}).get("accuracy"),
                    "official_evaluation_n": (row.get("label") or {}).get("evaluation_n"),
                    "recipe_fidelity_eligible_in_previous_audit": row["audit"]["eligible"],
                    "scope": "historical archive only; not current WM training membership",
                }
            )
        emit("historical_cards.json", json_bytes(card_index), "historical_archive_index")
        manifest = {
            "schema": "wm-disjoint-raw-history-v1",
            "evidence_root": "evidence",
            "files": dict(sorted(files.items())),
            "history_run_ids": sorted(train),
            "forbidden_candidate_run_ids": sorted(train),
            "intended_candidate_run_ids": sorted(test),
            "split_sha256": protocol["split_sha256"],
            "purpose": "identical disjoint historical reference for both selector arms; not WM features",
        }
        provenance = {
            "split_file_sha256": sha256(raw_split),
            "raw_index_sha256": sha256(raw_index),
            "inventory_sha256": sha256(raw_inventory),
            "source_revisions": sorted({r["source_revision"] for r in index}),
            "artifacts": artifacts,
            "limits": [
                "Credential redaction is heuristic, not a universal secret detector.",
                "Raw evidence retains outcomes of OTHER runs; never mount for a candidate from these runs.",
                "Official archive labels do not guarantee recipe/checkpoint fidelity.",
                "This is not a trained WM or an evaluation result.",
            ],
        }
        for name, value in (("provenance.json", provenance), ("manifest.json", manifest)):
            with (output / name).open("xb") as handle:
                handle.write(json_bytes(value))
            (output / name).chmod(0o444)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = package_history(args.dataset_dir, args.output_dir)
    print(
        json.dumps(
            {"files": len(result["files"]), "historical_runs": len(result["history_run_ids"])}
        )
    )


if __name__ == "__main__":
    main()
