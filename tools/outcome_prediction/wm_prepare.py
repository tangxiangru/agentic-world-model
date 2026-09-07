"""Freeze run-disjoint WM data and a learned artifact without making LLM calls.

Agent decision sets are a separate preparation stage: a train/test split alone
does not certify multi-candidate prompts against cross-candidate outcome leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from tools.outcome_prediction.wm_model import MODEL_SPEC, RecipeWorldModel, digest, public_payload

DEFAULT_SOURCES = [
    {
        "raw_root": "data/traj/raw/awm-gsm8k-trajectories",
        "manifest_name": "manifest_r0.json",
        "source_revision": "7294afde88f6e70bb7d0899de3e8e10f3172c622",
        "benchmark": "gsm8k",
        "base_model": "google/gemma-3-4b-pt",
        "evaluation_n": 1319,
    },
    {
        "raw_root": "data/traj/raw/awm-gsm8k-trajectories-88880836873e",
        "manifest_name": "manifest_aime_r0.json",
        "source_revision": "88880836873ebe5c351d7b0784ae53a171f4d659",
        "benchmark": "aime2025",
        "base_model": "Qwen/Qwen3-4B-Base",
        "evaluation_n": 30,
    },
]


def create_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def create_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_fingerprints():
    here = Path(__file__).parent
    return {
        name: hashlib.sha256((here / name).read_bytes()).hexdigest()
        for name in (
            "wm_prepare.py",
            "wm_dataset.py",
            "wm_model.py",
            "rpm_code_provenance.py",
            "build_examples.py",
        )
    }


def eligibility_summary(rows):
    result = {}
    for benchmark in sorted({r["benchmark"] for r in rows}):
        part = [r for r in rows if r["benchmark"] == benchmark]
        result[benchmark] = {
            "runs": len({r["cell_id"] for r in part}),
            "cards": len(part),
            "official_labels": sum(
                (r.get("label") or {}).get("accuracy") is not None for r in part
            ),
            "eligible_labeled": sum(
                r.get("audit", {}).get("eligible") is True
                and (r.get("label") or {}).get("accuracy") is not None
                for r in part
            ),
        }
    return result


def apply_previous_gsm_audit(rows, audit_path):
    """Keep known original recipe/label exclusions, with explicit provenance."""
    path = Path(audit_path)
    known = {r["example_id"]: r for r in map(json.loads, path.read_text().splitlines())}
    changed = []
    for row in rows:
        if row["benchmark"] != "gsm8k":
            continue
        if row["example_id"] not in known:
            raise ValueError("GSM example missing from the supplied prior audit")
        prior = known[row["example_id"]]
        if not prior.get("eligible"):
            row["audit"]["eligible"] = False
            row["audit"].setdefault("reasons", []).append(
                "excluded_by_previous_gsm_recipe_fidelity_audit"
            )
            row["audit"]["previous_gsm_audit"] = {
                "eligible": prior.get("eligible"),
                "exclusions": prior.get("exclusions", prior.get("exclusion_reasons")),
            }
            changed.append(row["example_id"])
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "excluded_ids": changed,
    }


def apply_fidelity_decisions(rows, path):
    path = Path(path)
    decisions = json.loads(path.read_text())
    by_id = {row["example_id"]: row for row in rows}
    for example_id, decision in decisions["decisions"].items():
        if example_id not in by_id or decision.get("action") not in {
            "exclude",
            "reviewed_no_exclusion",
        }:
            raise ValueError("Unknown ID or action in explicit fidelity decisions")
        row = by_id[example_id]
        row["audit"]["setup_change_review"] = decision
        if decision["action"] == "exclude":
            row["audit"]["eligible"] = False
            row["audit"].setdefault("reasons", []).append("audited_recipe_label_mismatch")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def normalize_eligibility(rows):
    for row in rows:
        row["initial_eligible"] = row.get(
            "initial_eligible", row.get("eligible", row["audit"]["eligible"])
        )
        row["eligible"] = row["audit"]["eligible"]


def add_known_checkpoint_history(rows):
    """Retrospective known official scores of already closed predecessors only.

    The grade's historical availability is not asserted. Multi-candidate packets
    must further remove every competing target node from this per-recipe state.
    """
    by_id = {row["example_id"]: row for row in rows}
    for row in rows:
        cutoff = row.get("first_submitted_at")
        latest = {}
        if cutoff:
            before = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
            for prior in row.get("prior_observations", []):
                previous = by_id.get(prior.get("example_id"))
                if not previous or previous["cell_id"] != row["cell_id"] or previous is row:
                    continue
                if (
                    prior.get("stage") != "closed"
                    or (previous.get("label") or {}).get("accuracy") is None
                ):
                    continue
                at = datetime.fromisoformat(prior["at"].replace("Z", "+00:00"))
                if at >= before:
                    raise ValueError("Future predecessor record in known-checkpoint context")
                old = latest.get(previous["example_id"])
                if old is None or at > old[0]:
                    latest[previous["example_id"]] = (at, prior, previous)
        row["model_input"]["known_previous_checkpoints"] = [
            {
                "checkpoint_ref": previous["card_id"],
                "official_accuracy": previous["label"]["accuracy"],
                "official_evaluation_n": previous["label"].get("evaluation_n"),
                "local_record_before_proposal": prior["record"],
                "score_availability": "retrospectively supplied official grade of an already closed predecessor",
            }
            for _, prior, previous in sorted(
                latest.values(), key=lambda item: (item[0], item[2]["example_id"])
            )
        ]


def prepare(
    *, sources, output_dir, test_count, seed, previous_gsm_audit=None, fidelity_decisions=None
):
    from tools.outcome_prediction.wm_dataset import extract_recorder_dataset, stratified_run_split

    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("Choose a new output directory; prepared datasets are immutable")
    code_hashes = source_fingerprints()
    rows, source_audits = [], []
    for source in sources:
        extracted, audit = extract_recorder_dataset(**source, recover_code=True)
        rows.extend(extracted)
        source_audits.append(audit)
    ids = [r["example_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Example IDs must be unique across sources")
    old_audit = apply_previous_gsm_audit(rows, previous_gsm_audit) if previous_gsm_audit else None
    fidelity_audit = (
        apply_fidelity_decisions(rows, fidelity_decisions) if fidelity_decisions else None
    )
    normalize_eligibility(rows)
    add_known_checkpoint_history(rows)
    split = stratified_run_split(rows, test_count=test_count, seed=seed)
    train_cells = set(split["train_cell_ids"])
    test_cells = set(split["test_cell_ids"])
    if train_cells & test_cells or train_cells | test_cells != {r["cell_id"] for r in rows}:
        raise ValueError("Incomplete or overlapping run partition")
    train_all = [r for r in rows if r["cell_id"] in train_cells]
    test_all = [r for r in rows if r["cell_id"] in test_cells]
    train = [
        r
        for r in train_all
        if r["audit"]["eligible"] and (r.get("label") or {}).get("accuracy") is not None
    ]
    test = [
        r
        for r in test_all
        if r["audit"]["eligible"] and (r.get("label") or {}).get("accuracy") is not None
    ]
    wm = RecipeWorldModel().fit(rows, train_cell_ids=train_cells, forbidden_cell_ids=test_cells)
    if code_hashes != source_fingerprints():
        raise ValueError("Source changed during preparation")
    root.mkdir(parents=True)
    create_jsonl(root / "private/inventory.jsonl", rows)
    create_jsonl(root / "train/records.jsonl", train)
    create_jsonl(
        root / "test/inputs.jsonl",
        [
            {
                "example_id": r["example_id"],
                "cell_id": r["cell_id"],
                "benchmark": r["benchmark"],
                "model_input": public_payload(r),
            }
            for r in test
        ],
    )
    create_jsonl(
        root / "private/test_labels.jsonl",
        [
            {"example_id": r["example_id"], "cell_id": r["cell_id"], "label": r["label"]}
            for r in test
        ],
    )
    create_json(root / "split.json", split)
    create_json(root / "source_audits.json", source_audits)
    create_json(root / "model/training_manifest.json", wm.training_manifest)
    model_hash = wm.save(root / "model/wm.joblib")
    for name, fingerprint in code_hashes.items():
        raw = (Path(__file__).parent / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != fingerprint:
            raise ValueError("Source changed during preparation")
        destination = root / "provenance/source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(raw)
    # Exact raw files are indexed here, not yet exposed to an agent. The later
    # evidence builder must verify hashes and credential-scan before publishing.
    trajectory_index = []
    for source in sources:
        raw_root = Path(source["raw_root"])
        manifest = json.loads((raw_root / source["manifest_name"]).read_text())
        for cell in manifest:
            if cell["cell_id"] not in train_cells:
                continue
            path = raw_root / "cells" / cell["cell_id"] / "solve_out_sanitized.txt"
            raw = path.read_bytes()
            trajectory_index.append(
                {
                    "cell_id": cell["cell_id"],
                    "benchmark": source["benchmark"],
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "source_revision": source["source_revision"],
                    "status": "training-run raw evidence indexed; exposure audit pending",
                }
            )
    create_json(root / "train/raw_trajectory_index.json", trajectory_index)
    protocol = {
        "version": 1,
        "sources": sources,
        "source_code_sha256": code_hashes,
        "split_sha256": digest(split),
        "model_spec": MODEL_SPEC,
        "model_sha256": model_hash,
        "previous_gsm_audit": old_audit,
        "fidelity_decisions": fidelity_audit,
        "counts": {
            "all": eligibility_summary(rows),
            "train": eligibility_summary(train_all),
            "test": eligibility_summary(test_all),
        },
        "raw_training_trajectories": len(trajectory_index),
        "training_payload_sha256": wm.training_manifest["training_data_sha256"],
        "training_runs_stratification": "benchmark and scientist model; identities only, not target scores",
        "test_outcomes_exposed_to_wm_fit": False,
        "test_input_policy": "Per-target forecasting packets, NOT a joint blinded choice set. Earlier test-run checkpoints can appear as known history for later targets. Never expose the complete test/inputs.jsonl to an agent; build a fresh shared-cutoff, all-candidates-blocked packet for each decision.",
        "safe_to_expose_whole_test_input_file": False,
        "known_checkpoint_score_policy": "Official grades of same-run predecessors with a closed record before the target proposal; retrospective known context, not claimed historical availability. Candidate-set filtering remains required.",
        "status": "run split frozen and provisional WM fitted; train target-fidelity and agent decision-packet audits pending",
        "evaluation_requirements": {
            "same_agent_and_budget_both_arms": True,
            "same_permitted_raw_trajectories_plans_code_history": True,
            "only_additional_capability": "fixed trained WM query tool",
            "primary_metric": "official accuracy of the checkpoint selected by each agent",
            "candidate_count_and_training_execution_interpretation": "awaiting user clarification of 15 recipe",
            "no_claim_of_fresh_unseen_release": "Previous studies inspected this release; fixed retrospective test only.",
        },
    }
    create_json(root / "protocol.json", protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources-json", type=Path)
    parser.add_argument("--test-count-per-stratum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--previous-gsm-audit",
        type=Path,
        default=Path("data/analysis/outcome_prediction/examples.jsonl"),
    )
    parser.add_argument("--fidelity-decisions", type=Path)
    args = parser.parse_args()
    sources = json.loads(args.sources_json.read_text()) if args.sources_json else DEFAULT_SOURCES
    protocol = prepare(
        sources=sources,
        output_dir=args.output_dir,
        test_count=args.test_count_per_stratum,
        seed=args.seed,
        previous_gsm_audit=args.previous_gsm_audit,
        fidelity_decisions=args.fidelity_decisions,
    )
    print(json.dumps({"counts": protocol["counts"], "status": protocol["status"]}, indent=2))


if __name__ == "__main__":
    main()
