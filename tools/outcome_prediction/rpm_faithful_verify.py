"""Verify frozen rich-RPM inputs without reading any model verdicts.

Reads immutable input/protocol/label metadata, original recorder cards, and the
manifest copy used for preparation. Writes verification.json only. Historical
measurement extraction and displayed-side ancestry are recomputed separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from tools.outcome_prediction.rpm_judge import digest
from tools.outcome_prediction.rpm_paper_prompt import render_prompt


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def descendants_of_reference(start, graph):
    """Follow upstream references, independent of stored ancestry membership."""
    found, pending = set(), list(graph.get(start, []))
    while pending:
        ref = pending.pop()
        if ref not in found:
            found.add(ref)
            pending.extend(graph.get(ref, []))
    return found - {"base_model"}


def normalized_text(text, cell, mapping):
    text = text.replace(cell, "current_run")
    return re.sub(
        r"(?i)(?<![A-Za-z0-9])exp[-_ ]?0*(\d+)(?!\d)",
        lambda match: mapping.get(f"exp-{int(match[1]):02d}", "unresolved_checkpoint"),
        text,
    )


def independent_local_score(card):
    values = []
    for measurement in (card.get("result") or {}).get("measurements") or []:
        name = str(measurement.get("name", "")) + " " + str(measurement.get("metric", ""))
        if not re.search("accuracy|gsm8k", name, re.IGNORECASE):
            continue
        try:
            value = float(measurement["value"])
            size = float(measurement.get("n") or 0)
        except (TypeError, ValueError, KeyError):
            continue
        if 1 < value <= 100:
            value /= 100
        if 0 <= value <= 1:
            values.append((size, value))
    if not values:
        return None
    maximum = max(size for size, _ in values)
    scores = {value for size, value in values if size == maximum}
    return next(iter(scores)) if len(scores) == 1 else None


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def verify(root, raw_root, examples):
    labels = json.loads((root / "hidden_labels.json").read_text())
    jobs = json.loads((root / "jobs.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    audits = {r["id"]: r for r in json.loads((root / "input_audit.json").read_text())}
    rows = {r["example_id"]: r for r in map(json.loads, examples.read_text().splitlines())}
    manifest_path = root / "code_provenance_manifest_used.json"
    manifest = json.loads(manifest_path.read_text())
    provenance = {r["example_id"]: r for r in manifest["cards"]}
    counts, errors, job_hashes = Counter(), [], []
    raw_hashes, code_hashes = {}, {}

    def check(test, identity, reason):
        if not test:
            errors.append({"id": identity, "reason": reason})

    def check_code(actual_scripts, eid, identity, cell, mapping, kind):
        entries = provenance[eid]["scripts"]
        check(len(actual_scripts) == len(entries), identity, "named script count mismatch")
        for actual, entry in zip(actual_scripts, entries):
            check(actual["role"] == entry["role"], identity, "script role mismatch")
            if entry["status"] != "reconstructed":
                check("content" not in actual, identity, "unavailable code transmitted")
                counts[kind + "_unavailable_code_blocks"] += 1
                continue
            path = Path(entry["reconstructed_content_file"])
            text = path.read_text()
            code_hashes[str(path)] = file_hash(path)
            check(
                hashlib.sha256(text.encode()).hexdigest() == entry["sha256"],
                identity,
                "code hash mismatch",
            )
            check(
                actual.get("content") == normalized_text(text, cell, mapping),
                identity,
                "code differs beyond ID substitution",
            )
            check(
                actual["script_path"] == normalized_text(entry["script_path"], cell, mapping),
                identity,
                "script path mismatch",
            )
            evidence = entry["evidence"]
            latest = max(timestamp(evidence["at"]), timestamp(evidence["envelope_at"]))
            check(
                latest < timestamp(entry["cutoff"]),
                identity,
                "code evidence not strictly before proposal",
            )
            check(
                entry["cutoff"] == rows[eid]["first_submitted_at"], identity, "code cutoff mismatch"
            )
            counts[kind + "_code_blocks"] += 1

    for label in labels:
        identity, cell = label["id"], label["cell_id"]
        packet = json.loads((root / "inputs" / (identity + ".json")).read_text())
        audit = audits[identity]
        mapping = {
            r["card_id"]: "checkpoint_" + digest(["node", cell, r["card_id"]])[:8]
            for r in rows.values()
            if r["cell_id"] == cell
        }
        inverse = {value: key for key, value in mapping.items()}
        a, b = label["a_id"], label["b_id"]
        displayed = [b, a] if label["swapped"] else [a, b]
        for name, eid in zip(("candidate_A", "candidate_B"), displayed, strict=True):
            candidate = packet[name]
            check(
                candidate["card_ref"] == mapping[rows[eid]["card_id"]],
                identity,
                "candidate orientation mismatch",
            )
            check(
                not (
                    {"result", "conclusion", "official_accuracy", "y", "y_a", "y_b"}
                    & set(candidate)
                ),
                identity,
                "candidate outcome field",
            )
            check(
                set(candidate["earliest_plan"]) <= {"problem", "hypothesis", "setup", "evaluation"},
                identity,
                "unapproved candidate plan section",
            )
            expected_parents = [mapping[x] for x in rows[eid]["parent_ids"]] or ["base_model"]
            check(
                candidate["weight_parent_refs"] == expected_parents,
                identity,
                "weight parent mismatch",
            )
            check(
                any(s["role"] == "training" and "content" in s for s in candidate["earliest_code"]),
                identity,
                "missing candidate training code",
            )
            check_code(candidate["earliest_code"], eid, identity, cell, mapping, "candidate")

        nodes = packet.get("scored_predecessors", []) + [
            packet["candidate_A"],
            packet["candidate_B"],
        ]
        weight_graph = {n["card_ref"]: n["weight_parent_refs"] for n in nodes}
        combined_graph = {
            n["card_ref"]: set(n["weight_parent_refs"]) | set(n["data_dependency_refs"])
            for n in nodes
        }
        for node in packet.get("scored_predecessors", []):
            cid = inverse[node["card_ref"]]
            eid = cell + "/" + cid
            counts["historical_nodes"] += 1
            check(
                cid not in audit["forbidden_card_ids"],
                identity,
                "candidate/future/descendant in history",
            )
            check(
                node["official_accuracy"] == rows[eid]["y"],
                identity,
                "official predecessor score mismatch",
            )
            for field, graph in (
                ("weight_ancestor_of", weight_graph),
                ("weight_or_data_ancestor_of", combined_graph),
            ):
                expected = {
                    name
                    for name in ("candidate_A", "candidate_B")
                    if node["card_ref"] in descendants_of_reference(packet[name]["card_ref"], graph)
                }
                check(set(node.get(field, [])) == expected, identity, field + " mismatch")
            records = []
            for path in (raw_root / rows[eid]["source_card"]).parent.glob("record-*.json"):
                records.append(json.loads(path.read_text()))
                raw_hashes[str(path)] = file_hash(path)
            available = [
                r
                for r in records
                if r["at"] < audit["cutoff"] and independent_local_score(r["card"]) is not None
            ]
            expected_score = (
                independent_local_score(max(available, key=lambda r: r["at"])["card"])
                if available
                else None
            )
            check(
                node["observed_local_accuracy"] == expected_score,
                identity,
                "local history mismatch",
            )
            check_code(node["earliest_code"], eid, identity, cell, mapping, "historical")
            audit_only = {
                "earliest_plan_source",
                "local_record_source",
                "earliest_plan_at",
                "local_record_at",
            }
            check(not (audit_only & set(node)), identity, "audit metadata in model input")

    for job in jobs:
        identity = job["id"]
        prompt = Path(job["prompt_path"]).read_text()
        packet = json.loads((root / "inputs" / (identity + ".json")).read_text())
        check(digest(prompt) == job["prompt_sha256"], identity, "prompt hash mismatch")
        check(digest(packet) == job["input_sha256"], identity, "input hash mismatch")
        check(
            prompt == render_prompt(packet, target="immediate"),
            identity,
            "rendered prompt mismatch",
        )
        check(
            not re.search(r"\br0-\d+\b|\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{25,})", prompt),
            identity,
            "run ID or credential pattern",
        )
        job_hashes.append(
            {"id": identity, "prompt_sha256": digest(prompt), "input_sha256": digest(packet)}
        )
        counts["jobs_checked"] += 1
        if job.get("swap_of"):
            original = json.loads((root / "inputs" / (job["swap_of"] + ".json")).read_text())
            original["candidate_A"], original["candidate_B"] = (
                original["candidate_B"],
                original["candidate_A"],
            )
            for node in original.get("scored_predecessors", []):
                for field in ("weight_ancestor_of", "weight_or_data_ancestor_of"):
                    node[field] = [
                        {"candidate_A": "candidate_B", "candidate_B": "candidate_A"}[x]
                        for x in node.get(field, [])
                    ]
            check(original == packet, identity, "swap input mismatch")

    check(protocol["pairs"] == len(labels), "protocol", "cohort size mismatch")
    check(
        protocol["code_provenance_sha256"] == digest(manifest),
        "protocol",
        "frozen manifest mismatch",
    )
    check(
        not (set(protocol.get("semantic_exclusions", {})) & {p["id"] for p in labels}),
        "protocol",
        "excluded pair present",
    )
    return {
        "schema": "rpm-faithful-independent-verification-v1",
        "passed": not errors,
        "main_pairs": len(labels),
        "cells": len({p["cell_id"] for p in labels}),
        "known_nonbase_parent_pairs": sum(p["has_nonbase_parent"] for p in labels),
        "strict_unredacted_pairs": sum(p["full_plan_strict_eligible"] for p in labels),
        "counts": dict(counts),
        "errors": errors,
        "protocol_sha256": digest(protocol),
        "frozen_manifest_sha256": digest(manifest),
        "artifact_file_sha256": {
            str(path): file_hash(path)
            for path in (
                root / "protocol.json",
                root / "jobs.json",
                root / "hidden_labels.json",
                root / "input_audit.json",
                manifest_path,
                examples,
            )
        },
        "source_file_sha256": {
            str(path): file_hash(path) for path in sorted(Path(__file__).parent.glob("rpm_*.py"))
        },
        "jobs": job_hashes,
        "raw_record_sha256": raw_hashes,
        "code_file_sha256": code_hashes,
        "scope": "No model outputs or performance statistics read. This verifies provenance and metadata, not a guarantee against arbitrary unlogged execution or every implicit semantic clue.",
        "semantic_review": {
            "region": "r0-01 through r0-14, retained candidate narratives and core setup",
            "additional_exclusions_reported": [
                "pair-53f2d01d49c8: indirect MetaMath regression clue",
                "pair-39526460a016: implicit stopping-still-broken clue",
                "pair-e6b162fde5bb: candidate data generated from other candidate weights",
                "pair-1cf177d1e867: candidate data generated from other candidate weights",
            ],
            "other_regions": "Reviewed by separate pipeline/architecture agents; not independently recertified by this script.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/analysis/rpm/faithful"))
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/traj/raw/awm-gsm8k-trajectories")
    )
    parser.add_argument(
        "--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl")
    )
    args = parser.parse_args()
    report = verify(args.root, args.raw_root, args.examples)
    destination = args.root / "verification.json"
    destination.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("passed", "main_pairs", "cells", "counts", "errors", "protocol_sha256")
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
