"""Audit and prepare rich earliest-plan packets with known predecessor scores.

No model calls. Candidate outcomes and their descendants are never context.
Official predecessor accuracies are explicitly retrospective knowledge, not
claims that these measurements were available at the historical proposal time.
Structural outcome sections are omitted. A blocked reference removes an entire
problem/hypothesis/evaluation section; blocked setup narrative removes its entire
leaf. Core setup dependencies are rejected. Hyperparameter numbers are preserved.

build_pair(a, b, all_rows, raw_root) returns (payload_or_None, audit).
An accepted sanitized packet is not necessarily strict-unredacted eligible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from functools import cache
from pathlib import Path

from tools.outcome_prediction.rpm_judge import history_score

PLAN_FIELDS = ("problem", "hypothesis", "setup", "evaluation")
OBSERVED = re.compile(
    r"\b(?:scored|reached|achieved|collapsed|regressed|failed|underperformed|"
    r"outperformed|improved|worsened|was worse|was better|performed worse|performed better)\b",
    re.IGNORECASE,
)
IMPLICIT = re.compile(
    r"\b(?:incumbent|current best|previous (?:run|model|checkpoint|experiment)|"
    r"earlier (?:run|model|checkpoint|experiment)|last (?:run|model|checkpoint|experiment))\b",
    re.IGNORECASE,
)
SCORE_STATEMENT = re.compile(
    r"\b(?:accuracy|score|acc|pass@\d)\s*(?:is|was|of|=|:|at|reached)?\s*"
    r"(?:0?\.\d+|\d+(?:\.\d+)?\s*%)",
    re.IGNORECASE,
)
BASE = re.compile(
    r"\b(?:base model|base checkpoint|pretrained|pre-trained|untrained)\b", re.IGNORECASE
)
GENERIC_ALIASES = {
    "final",
    "final_model",
    "model",
    "checkpoint",
    "ckpts",
    "checkpoints",
    "task",
    "results",
    "result",
    "evals",
    "eval",
    "logs",
    "log",
    "runs",
    "outputs",
    "output",
}

# Reviewed against each earliest setup.data record. These references describe
# shared pre-existing inputs or processing rules, not samples from that node's
# newly trained weights. Keep them explicit rather than silently conflating data
# prose references with actual dependency edges.
INFORMATIONAL_DATA_REFS = {
    (
        "r0-01/exp-03",
        "exp-02",
    ): "Reuses exp-02's input mixture; its generated samples originate from exp-01.",
    (
        "r0-06/exp-04",
        "exp-03",
    ): "Reuses samples generated from exp-02, previously prepared for exp-03.",
    (
        "r0-08/exp-05",
        "exp-01",
    ): "Same external OpenMathInstruct input file, not exp-01-generated samples.",
    (
        "r0-27/exp-03",
        "exp-02",
    ): "Same cleaning rules; both datasets come directly from GSM8K/MetaMath.",
    ("r0-28/exp-06", "exp-04"): "Reweights external OpenMathInstruct rows also seen by exp-04.",
    (
        "r0-30/exp-06",
        "exp-05",
    ): "Filtering fix for exp-05; generator and replay inputs originate from exp-03.",
    (
        "r0-32/exp-02",
        "exp-01",
    ): "Same cleaning rules applied to a larger external OpenMathInstruct split.",
}

# The generic "RFT rounds 1+2" source omits checkpoint names. The reviewed
# pooled trace files are sft_out_e3 (exp-01) and sft_out_rft (exp-02).
VERIFIED_DATA_EDGES = {"r0-11/exp-07": {"exp-01", "exp-02"}}


@cache
def read_records(raw_root: str, source_card: str) -> tuple:
    folder = (Path(raw_root) / source_card).parent
    records = [(json.loads(p.read_text()), p) for p in folder.glob("record-*.json")]
    if not records:
        raise ValueError(f"No records for {source_card}")
    return tuple(
        {"at": r["at"], "card": r["card"], "source": str(p)}
        for r, p in sorted(records, key=lambda item: (item[0]["at"], item[1].name))
    )


def read_first(raw_root: str, source_card: str) -> dict:
    return read_records(raw_root, source_card)[0]


def card_references(text: str) -> set[str]:
    result = set()
    for match in re.finditer(r"(?i)(?<![A-Za-z0-9])exp[-_ ]?0*(\d+)((?:/0*\d+)*)", text):
        result.add(f"exp-{int(match[1]):02d}")
        result.update(f"exp-{int(value):02d}" for value in match[2].split("/") if value)
    return result


def leaves(obj, prefix=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, (*prefix, str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from leaves(value, (*prefix, str(i)))
    elif isinstance(obj, str):
        yield prefix, obj


def drop_path(obj, path):
    current = obj
    for key in path[:-1]:
        current = current[int(key)] if isinstance(current, list) else current[key]
    if isinstance(current, list):
        current[int(path[-1])] = None
    else:
        current.pop(path[-1], None)


def data_edges(row, first, references=card_references):
    inferred = set(row.get("data_dependency_ids", [])) | VERIFIED_DATA_EDGES.get(
        row["example_id"], set()
    )
    # Source fields catch compact references such as derived:exp-01/02, which
    # the original extraction's single-card pattern did not expand.
    for data in first.get("setup", {}).get("data", []):
        inferred |= references(str(data.get("source", "")))
        inferred |= references(str(data.get("selection", "")))
        command = data.get("build_command") or []
        if isinstance(command, str):
            command = command.split()
        for i, token in enumerate(command[:-1]):
            if re.fullmatch(
                r"--(?:model(?:[-_]path)?|teacher|checkpoint|generator|scorer)", str(token)
            ):
                inferred |= references(str(command[i + 1]))
    informational = {
        ref: INFORMATIONAL_DATA_REFS[(row["example_id"], ref)]
        for ref in inferred
        if (row["example_id"], ref) in INFORMATIONAL_DATA_REFS
    }
    return inferred - set(informational) - {row["card_id"]}, informational


def ancestor_ids(start, graph):
    seen, pending = set(), list(graph.get(start, []))
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(graph.get(node, []))
    return seen


def narrative_path(path):
    return path[-1] in {
        "other",
        "selection",
        "target_format",
        "notes",
        "note",
        "description",
    } or path[:2] == ("method", "description")


def build_pair(a: dict, b: dict, all_rows, raw_root: Path):
    """Return a structurally sanitized packet and a separate, non-prompt audit."""
    if a["cell_id"] != b["cell_id"] or a["example_id"] == b["example_id"]:
        raise ValueError("Distinct candidates in the same cell are required")
    iterable = all_rows.values() if isinstance(all_rows, dict) else all_rows
    rows = {r["card_id"]: r for r in iterable if r["cell_id"] == a["cell_id"]}
    first = {cid: read_first(str(raw_root), row["source_card"]) for cid, row in rows.items()}
    weight = {cid: set(row.get("parent_ids", [])) for cid, row in rows.items()}
    candidates = {a["card_id"], b["card_id"]}
    cutoff = min(first[cid]["at"] for cid in candidates)
    aliases = {}
    for cid, row in rows.items():
        setup = first[cid]["card"].get("setup", {})
        values = [
            setup.get("output_dir"),
            (first[cid]["card"].get("result") or {}).get("output_checkpoint"),
        ]
        for value in values:
            if not isinstance(value, str):
                continue
            name = value.rstrip("/").rsplit("/", 1)[-1]
            if name.lower() in GENERIC_ALIASES:
                continue
            aliases.setdefault(value.rstrip("/"), set()).add(cid)
            if len(name) >= 3:
                aliases.setdefault(name, set()).add(cid)

    def references(text):
        refs = card_references(text) & set(rows)
        for alias, ids in aliases.items():
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_/])", text):
                refs |= ids
        return refs

    data, informational = {}, {}
    for cid, row in rows.items():
        data[cid], informational[cid] = data_edges(row, first[cid]["card"], references)
    combined = {cid: weight[cid] | data[cid] for cid in rows}
    descendants = {cid for cid in rows if ancestor_ids(cid, combined) & candidates}
    future = {cid for cid in rows if first[cid]["at"] >= cutoff} - candidates
    forbidden = candidates | descendants | future
    previous = set(rows) - forbidden

    audit = {
        "accepted": False,
        "strict_unredacted": False,
        "a_id": a["example_id"],
        "b_id": b["example_id"],
        "cell_id": a["cell_id"],
        "reasons": [],
        "redactions": [],
        "structural_exclusions": ["result", "conclusion", "other top-level metadata"],
        "forbidden_card_ids": sorted(forbidden),
        "descendant_card_ids": sorted(descendants),
        "cutoff": cutoff,
        "data_dependency_review": {},
    }
    for me, other in ((a, b), (b, a)):
        cid = me["card_id"]
        audit["data_dependency_review"][cid] = {
            "weight_parent_ids": sorted(weight[cid]),
            "data_dependency_ids": sorted(data[cid]),
            "informational_data_references": informational[cid],
        }
        if other["card_id"] in ancestor_ids(cid, combined):
            audit["reasons"].append(
                f"{cid}: candidate depends on the other candidate's weights/data"
            )
    if audit["reasons"]:
        return None, audit

    packets = {}
    for name, row in (("candidate_A", a), ("candidate_B", b)):
        cid = row["card_id"]
        prior_forbidden = {ref for ref in forbidden - {cid} if first[ref]["at"] < first[cid]["at"]}
        plan = {
            key: copy.deepcopy(first[cid]["card"][key])
            for key in PLAN_FIELDS
            if key in first[cid]["card"]
        }
        for section in ("problem", "hypothesis", "evaluation"):
            reasons = []
            section_refs = set().union(
                *(references(text) for _, text in leaves(plan.get(section, {})))
            )
            section_attributed = bool(section_refs & previous) and not (section_refs & forbidden)
            for path, text in leaves(plan.get(section, {})):
                refs = references(text)
                blocked = refs & (forbidden - {cid})
                if blocked:
                    reasons.append(
                        {
                            "path": ".".join(path),
                            "kind": "blocked_reference",
                            "refs": sorted(blocked),
                        }
                    )
                elif cid in refs and OBSERVED.search(text):
                    reasons.append({"path": ".".join(path), "kind": "own_observed_outcome"})
                elif (
                    prior_forbidden
                    and not section_attributed
                    and not refs
                    and not BASE.search(text)
                    and (
                        IMPLICIT.search(text)
                        or OBSERVED.search(text)
                        or SCORE_STATEMENT.search(text)
                    )
                ):
                    reasons.append(
                        {"path": ".".join(path), "kind": "unattributed_outcome_or_incumbent_clue"}
                    )
            # A comparator value is meaningful only with an explicitly permitted
            # node or base-model reference; preserve protocol hyperparameters.
            if section == "evaluation":
                comparator = plan.get(section, {}).get("comparator") or {}
                if comparator.get("value") is not None:
                    target = (
                        str(comparator.get("ref") or "") + " " + str(comparator.get("path") or "")
                    )
                    if (
                        prior_forbidden
                        and not (references(target) & previous)
                        and not BASE.search(target)
                        and str(comparator.get("ref")) not in {"base", "base_model"}
                    ):
                        reasons.append(
                            {"path": "comparator", "kind": "unresolved_or_blocked_comparator_score"}
                        )
            if reasons:
                plan.pop(section, None)
                audit["redactions"].append(
                    {
                        "candidate": cid,
                        "path": section,
                        "scope": "whole_section",
                        "findings": reasons,
                    }
                )
        for path, text in list(leaves(plan.get("setup", {}))):
            refs = references(text)
            blocked = refs & (forbidden - {cid})
            # A reviewed reference to a shared external input is not an outcome
            # and must not erase the actual filtering/masking recipe.
            if not (OBSERVED.search(text) or SCORE_STATEMENT.search(text)):
                blocked -= set(informational[cid])
            ambiguous = (
                prior_forbidden
                and not refs
                and not BASE.search(text)
                and (IMPLICIT.search(text) or SCORE_STATEMENT.search(text))
            )
            own_outcome = cid in refs and OBSERVED.search(text)
            if not (blocked or ambiguous or own_outcome):
                continue
            finding = {
                "candidate": cid,
                "path": "setup." + ".".join(path),
                "refs": sorted(blocked),
                "scope": "whole_narrative_leaf",
            }
            if narrative_path(path):
                drop_path(plan["setup"], path)
                audit["redactions"].append(finding)
            else:
                audit["reasons"].append(
                    f"{cid}: blocked/unattributed outcome in core setup at {'.'.join(path)}"
                )
        packets[name] = {
            "card_ref": cid,
            "earliest_plan": plan,
            "weight_parent_refs": sorted(weight[cid]) or ["base_model"],
            "data_dependency_refs": sorted(data[cid]),
            "informational_data_references": sorted(informational[cid]),
        }
    if audit["reasons"]:
        return None, audit
    weight_a, weight_b = ancestor_ids(a["card_id"], weight), ancestor_ids(b["card_id"], weight)
    ancestry_a, ancestry_b = (
        ancestor_ids(a["card_id"], combined),
        ancestor_ids(b["card_id"], combined),
    )
    predecessors = []
    for cid in sorted(previous):
        row = rows[cid]
        observed = [
            record
            for record in read_records(str(raw_root), row["source_card"])
            if record["at"] < cutoff and history_score(record["card"]) is not None
        ]
        record = observed[-1] if observed else None
        known_parent = cid in ancestry_a | ancestry_b
        if record is None and not known_parent:
            continue
        predecessors.append(
            {
                "card_ref": cid,
                "recipe": row["recipe"],
                "weight_parent_refs": sorted(weight[cid]) or ["base_model"],
                "data_dependency_refs": sorted(data[cid]),
                "official_accuracy": row.get("y"),
                "measurement_scope": "retrospectively known archived immediate checkpoint accuracy",
                "observed_local_accuracy": history_score(record["card"]) if record else None,
                "local_measurement_scope": "scientist-reported local evaluation before proposal cutoff; subset/protocol may vary",
                "local_record_at": record["at"] if record else None,
                "local_record_source": record["source"] if record else None,
                "earliest_plan_at": first[cid]["at"],
                "earliest_plan_source": first[cid]["source"],
                "weight_or_data_ancestor_of": [
                    name
                    for name, ancestors in (
                        ("candidate_A", ancestry_a),
                        ("candidate_B", ancestry_b),
                    )
                    if cid in ancestors
                ],
                "weight_ancestor_of": [
                    name
                    for name, ancestors in (("candidate_A", weight_a), ("candidate_B", weight_b))
                    if cid in ancestors
                ],
            }
        )
    audit["accepted"] = True
    audit["strict_unredacted"] = not audit["redactions"]
    payload = {
        "setting": "retrospective known-predecessor immediate-checkpoint ranking",
        "score_policy": "Only preceding noncandidate, nondescendant nodes are scored. Official scores are retrospective knowledge, not historical availability claims.",
        **packets,
        "common_weight_ancestor_refs": sorted(weight_a & weight_b) or ["base_model"],
        "scored_predecessors": predecessors,
        "base_checkpoint": {"model": a["recipe"].get("base_model"), "official_accuracy": None},
        "omitted_fields": [
            {"candidate": item["candidate"], "path": item["path"]} for item in audit["redactions"]
        ],
    }
    return payload, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl")
    )
    parser.add_argument(
        "--pairs", type=Path, default=Path("data/analysis/rpm/judge/hidden_labels.json")
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/traj/raw/awm-gsm8k-trajectories")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/analysis/rpm/rich_context_v2")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(
            "Use a fresh output directory so stale accepted packets cannot survive an audit change"
        )
    rows = [json.loads(line) for line in args.examples.read_text().splitlines() if line.strip()]
    by_id = {row["example_id"]: row for row in rows}
    audits = []
    for pair in json.loads(args.pairs.read_text()):
        payload, audit = build_pair(by_id[pair["a_id"]], by_id[pair["b_id"]], rows, args.raw_root)
        audit["pair_id"] = pair["id"]
        if payload is not None:
            path = args.output_dir / "sanitized" / (pair["id"] + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            path.write_text(text)
            audit["payload_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            if audit["strict_unredacted"]:
                strict = args.output_dir / "strict" / path.name
                strict.parent.mkdir(parents=True, exist_ok=True)
                strict.write_text(text)
        audits.append(audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(audits, indent=2, sort_keys=True) + "\n")
    accepted = [a for a in audits if a["accepted"]]
    strict = [a for a in audits if a["strict_unredacted"]]
    coverage = {
        "total_pairs": len(audits),
        "sanitized_pairs": len(accepted),
        "strict_pairs": len(strict),
        "sanitized_cells": len({a["cell_id"] for a in accepted}),
        "strict_cells": len({a["cell_id"] for a in strict}),
        "rejection_reasons": dict(Counter(reason for a in audits for reason in a["reasons"])),
        "manual_semantic_review_required_before_inference": True,
    }
    (args.output_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
