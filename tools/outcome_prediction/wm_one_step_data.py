"""Archive-correlated, labeled-only recipe data with an observed-parent subset.

Never executes scientist code, imputes accuracy, or selects on score magnitude.
Recorder archives are write-once per card; the first successful archive event,
not the last mutable card, identifies what the official card metric evaluates.
Exported traces do not certify checkpoint bytes or the full executed program.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import posixpath
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from tools.outcome_prediction.wm_clean_refresh import (
    OUTPUT as PREVIOUS_DATA,
)
from tools.outcome_prediction.wm_clean_refresh import (
    REVIEW_HOLDS,
    load_jsonl,
    project_example,
    read,
    sha,
    write,
)
from tools.outcome_prediction.wm_grade_inventory import valid_official_label

OUTPUT = Path("data/analysis/wm_one_step/01406da734fb_v1")
REQUIRED_FILES = {
    "private/registry.json", "private/decisions.json", "target_inputs.json",
    "target_labels.json", "one_step_inputs.json", "one_step_labels.json",
    "split.json", "summary.json", "policy.json",
}
ARCHIVE_REVIEW_HOLDS = {
    "aime2-r0-21/exp-02": {
        "reason": "archived_config_modified_after_capture_before_official_grade",
        "trace_sha256": "cc0b86a86940f152c3b49d68ded76c5253611e82d2340a13675fb340235cac00",
        "trace_line": 5685,
        "call_line_sha256": "fb00040165f4586b7dc48d82fe3627d05deb264725266a1d23c7e7e6355caa1a",
        "at": "2026-09-05T23:19:57Z",
    },
}


def stamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else None
    except (ValueError, TypeError, AttributeError):
        return None


def path_key(value, cwd="/home/ben/task"):
    if not isinstance(value, str) or not value.strip():
        return None
    if re.search(r"[$`{}*?\n\r]", value) or value.startswith("~"):
        return None
    if not isinstance(cwd, str) or not cwd.startswith("/"):
        return None
    return posixpath.normpath(posixpath.join(cwd, value.strip()))


def verified_file(provenance, name):
    path = Path(provenance[name + "_path"])
    if sha(path) != provenance[name + "_sha256"]:
        raise ValueError(f"Changed raw source: {path}")
    return path


def archive_binding(row):
    """Return score provenance independently of prospective-recipe eligibility."""
    provenance = row["provenance"]
    result = {
        "example_id": row["example_id"],
        "cell_id": row["cell_id"],
        "card_id": row["card_id"],
        "benchmark": row["benchmark"],
        "status": "unresolved",
        "reasons": [],
        "byte_identity_certified": False,
    }
    ledger_path = verified_file(provenance, "ledger")
    card_path = verified_file(provenance, "card")
    if provenance.get("first_record_path"):
        verified_file(provenance, "first_record")
    ledger = load_jsonl(ledger_path)
    events = [
        (line, event)
        for line, event in enumerate(ledger, 1)
        if event.get("card_id") == row["card_id"] and event.get("archived")
    ]
    if not events:
        result["reasons"].append("no_archive_event")
        return result
    line, event = events[0]
    number = event.get("record_n")
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        result["reasons"].append("invalid_archive_record_number")
        return result
    record_path = card_path.parent / f"record-{number:02d}.json"
    if not record_path.is_file():
        result["reasons"].append("missing_archive_record")
        return result
    record = read(record_path)
    card = record.get("card") or {}
    output = card.get("result") or {}
    archived = path_key(event.get("archived"))
    if not archived or not archived.endswith("/wm/checkpoints/" + row["card_id"]):
        result["reasons"].append("unexpected_archive_path")
        return result
    session = archived[: -len("/wm/checkpoints/" + row["card_id"])]
    source = path_key(output.get("output_checkpoint"), session)
    if not source or not source.startswith(session + "/"):
        result["reasons"].append("invalid_archive_source_path")
    if (
        event.get("event") != "submit"
        or record.get("event") != "submit"
        or card.get("card_id") != row["card_id"]
        or output.get("execution") != "completed"
        or event.get("stage") == "plan"
        or path_key(event.get("path"))
        != session + "/wm/cards/" + row["card_id"] + "/" + record_path.name
    ):
        result["reasons"].append("archive_record_identity_mismatch")
    at, record_at = stamp(event.get("at")), stamp(record.get("at"))
    if at is None or record_at is None or record_at > at:
        result["reasons"].append("invalid_archive_time")
    if any(path_key(e.get("archived")) != archived for _, e in events):
        result["reasons"].append("conflicting_archive_paths")
    # Retain the original archived setup only in the audit registry.
    result.update(
        source_path=source,
        archive_path=archived,
        archived_at=event.get("at"),
        archived_record_at=record.get("at"),
        archived_setup=copy.deepcopy(card.get("setup")),
        latest_card_output=path_key((read(card_path).get("result") or {}).get("output_checkpoint")),
        evidence={
            "ledger_path": str(ledger_path),
            "ledger_sha256": sha(ledger_path),
            "ledger_line": line,
            "record_path": str(record_path),
            "record_sha256": sha(record_path),
            "record_n": number,
        },
    )
    if result["reasons"]:
        return result
    result["status"] = "archive_without_valid_grade"
    if not valid_official_label(row.get("label")):
        result["reasons"].append("missing_or_invalid_official_label")
        return result
    metric_path = verified_file(provenance, "official_metric")
    if read(metric_path) != row["label"]["official_metric"]:
        raise ValueError("Metric content differs from frozen inventory")
    if metric_path != ledger_path.parent.parent / "wm_metrics" / (row["card_id"] + ".json"):
        result["reasons"].append("metric_card_identity_mismatch")
        return result
    log_path = ledger_path.parent.parent / "output.log"
    needle = "card checkpoint evaluation: " + row["card_id"] + " ("
    lines = [
        i for i, text in enumerate(log_path.read_text().splitlines(), 1) if needle in text
    ] if log_path.is_file() else []
    result["evidence"].update(
        metric_path=str(metric_path),
        metric_sha256=sha(metric_path),
        output_log_path=str(log_path) if log_path.is_file() else None,
        output_log_sha256=sha(log_path) if log_path.is_file() else None,
        output_log_lines=lines,
        metric_binding_basis="recorder_archive_plus_per_card_official_metric_export_contract",
        explicit_harness_evaluation_marker=bool(lines),
    )
    result.update(
        status="official_archive_correlated",
        accuracy=row["label"]["accuracy"],
        evaluation_n=row["label"]["evaluation_n"],
    )
    if row["example_id"] in ARCHIVE_REVIEW_HOLDS:
        # A changed source does not silently clear an explicit integrity hold.
        result["status"] = "unresolved"
        result["reasons"].append("known_archive_integrity_hold")
        result["integrity_review"] = copy.deepcopy(ARCHIVE_REVIEW_HOLDS[row["example_id"]])
    return result


def target_decision(row, archived, resolver):
    reasons = []
    audit = row.get("audit") or {}
    reasons.extend(code_provenance_issues(row))
    if not valid_official_label(row.get("label")):
        reasons.append("missing_or_invalid_official_label")
    if row.get("eligible") is not True or audit.get("eligible") is not True:
        reasons.append("failed_structural_screen")
    reasons.extend("structural:" + str(r) for r in audit.get("reasons", []))
    if row.get("first_stage") != "plan":
        reasons.append("first_submission_not_plan")
    if stamp(row.get("first_submitted_at")) is None:
        reasons.append("invalid_proposal_time")
    if audit.get("first_missing_fields"):
        reasons.append("incomplete_first_registration")
    if row["example_id"] in REVIEW_HOLDS:
        reasons.append("unresolved_prior_target_binding_hold")
    if archived["status"] != "official_archive_correlated":
        reasons.append("unresolved_official_target_archive")
    proposal, archive_time = stamp(row.get("first_submitted_at")), stamp(archived.get("archived_at"))
    if proposal is None or archive_time is None or proposal >= archive_time:
        reasons.append("archive_not_after_proposal")
    initial_setup = ((row.get("model_input") or {}).get("plan") or {}).get("setup")
    archived_setup = archived.get("archived_setup")
    changes = []
    if not isinstance(initial_setup, dict) or not isinstance(archived_setup, dict):
        reasons.append("missing_setup_for_archive_comparison")
    else:
        changes = sorted(
            key for key in initial_setup.keys() | archived_setup.keys()
            if (key in initial_setup) != (key in archived_setup)
            or initial_setup.get(key) != archived_setup.get(key)
        )
        if changes:
            reasons.append("proposal_to_archived_setup_changed")
    # The resolver sees the FIRST archived source, never a later substituted output.
    compared = copy.deepcopy(row)
    compared.setdefault("audit", {})["final_output_checkpoint"] = archived.get("source_path")
    compared["audit"].setdefault("output_artifact_comparison", {})[
        "final_output_checkpoint"
    ] = archived.get("source_path")
    binding = resolver(compared)
    if binding["status"] not in {"exact", "planned_save_path"}:
        reasons.append("unresolved_recipe_to_scored_output")
    return {
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "proposal_to_archived_setup_changed_fields": changes,
        "target_binding": binding,
        "runtime_certified": False,
        "proposal_at": row.get("first_submitted_at"),
    }


def code_provenance_issues(row):
    """Bind statically inspected code to the pre-proposal reconstruction evidence."""
    issues = []
    before = stamp(row.get("first_submitted_at"))
    evidence = (row.get("audit") or {}).get("code_provenance") or []
    for code in (row.get("model_input") or {}).get("code") or []:
        if code.get("status") != "reconstructed":
            continue
        matches = [e for e in evidence if e.get("role") == code.get("role")
                   and e.get("script_path") == code.get("script_path")]
        if len(matches) != 1 or not isinstance(code.get("content"), str):
            issues.append("code_reconstruction_provenance_missing")
            continue
        proof = matches[0]
        version = proof.get("evidence") or {}
        times = [stamp(version.get(k)) for k in ("at", "envelope_at")]
        if (
            proof.get("status") != "reconstructed"
            or proof.get("sha256") != hashlib.sha256(code["content"].encode()).hexdigest()
            or stamp(proof.get("cutoff")) != before
            or before is None or any(t is None or t >= before for t in times)
            or proof.get("blockers")
        ):
            issues.append("code_not_bound_to_preproposal_version")
    return sorted(set(issues))


def parent_reference(row, inputs, registry):
    """A single unique, earlier, same-session/protocol, exactly named archive."""
    answer = {"status": "unresolved", "reasons": [], "reference_known": False}
    if inputs["status"] == "base":
        answer["reasons"] = ["base_official_protocol_score_unverified"]
        return answer
    if inputs["status"] == "multiple":
        answer["reasons"] = ["multiple_checkpoint_inputs_not_a_single_parent_delta"]
        return answer
    if inputs["status"] != "single" or len(inputs["inputs"]) != 1:
        answer["reasons"] = ["checkpoint_input_unresolved"]
        return answer
    target = path_key(inputs["inputs"][0].get("path"))
    proposal = stamp(row.get("first_submitted_at"))
    if not target or proposal is None:
        answer["reasons"] = ["invalid_input_path_or_proposal_time"]
        return answer
    matching = [
        item for item in registry.values()
        if item["cell_id"] == row["cell_id"]
        and item["example_id"] != row["example_id"]
        and target in {item.get("source_path"), item.get("archive_path")}
    ]
    candidates = [item for item in matching if parent_precedes_proposal(item, row)]
    if len(candidates) != 1:
        answer["reasons"] = [
            "ambiguous_reused_checkpoint_path" if candidates else (
                "matching_archive_not_proven_before_proposal" if matching else "no_exact_scored_archive"
            )
        ]
        answer["candidate_ids"] = [c["example_id"] for c in matching]
        answer["candidate_archive_times"] = {c["example_id"]: c.get("archived_at") for c in matching}
        return answer
    parent = candidates[0]
    if parent["status"] != "official_archive_correlated":
        answer["reasons"] = [
            "parent_archive_integrity_hold" if "known_archive_integrity_hold" in parent["reasons"]
            else "parent_archive_has_no_valid_official_grade" if parent["status"] == "archive_without_valid_grade"
            else "parent_archive_binding_unresolved"
        ]
        return answer
    if parent["benchmark"] != row["benchmark"] or parent["evaluation_n"] != row["label"]["evaluation_n"]:
        answer["reasons"] = ["parent_child_evaluation_protocol_mismatch"]
        return answer
    answer.update(
        status="official_archive_correlated",
        reference_known=True,
        producer_id=parent["example_id"],
        consumed_path=target,
        path_kind="immutable_archive" if target == parent["archive_path"] else "archived_source",
        accuracy=parent["accuracy"],
        evaluation_n=parent["evaluation_n"],
        archived_at=parent["archived_at"],
        timing_basis=(
            "same_second_ordered_recorder_events"
            if stamp(parent["archived_at"]) == proposal else "strictly_earlier_timestamp"
        ),
        evidence=copy.deepcopy(parent["evidence"]),
        byte_identity_certified=False,
        official_grade_availability="retrospective_observed_parent_assumption",
    )
    return answer


def parent_precedes_proposal(parent, child):
    """Use ledger sequence to disambiguate equal second-resolution timestamps."""
    at, proposed = stamp(parent.get("archived_at")), stamp(child.get("first_submitted_at"))
    if at is None or proposed is None:
        return False
    if at != proposed:
        return at < proposed
    provenance = child.get("provenance") or {}
    if not provenance.get("ledger_path") or not provenance.get("first_record_path"):
        return False
    ledger_path = verified_file(provenance, "ledger")
    if str(ledger_path) != (parent.get("evidence") or {}).get("ledger_path"):
        return False
    first_name = Path(provenance["first_record_path"]).name
    events = [
        (line, event) for line, event in enumerate(load_jsonl(ledger_path), 1)
        if event.get("card_id") == child["card_id"] and event.get("event") == "submit"
        and Path(event.get("path") or "").name == first_name
    ]
    return bool(
        len(events) == 1
        and (parent.get("evidence") or {}).get("ledger_line", float("inf")) < events[0][0]
        and events[0][1].get("stage") == "plan"
        and stamp(events[0][1].get("at")) == proposed
    )


def build(rows, split, target_resolver, input_resolver, *, previous_rows=(), mutation_resolver=None):
    if mutation_resolver is None:
        from tools.outcome_prediction.wm_checkpoint_mutations import audit_source_use

        mutation_resolver = audit_source_use
    by_id = {r["example_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate example identity")
    if any(split["cell_partition"].get(r["cell_id"]) not in {"train", "test"} for r in rows):
        raise ValueError("Every session must have a fixed partition")
    registry = {key: archive_binding(row) for key, row in by_id.items()}
    previous = {r["example_id"]: r for r in previous_rows}
    decisions, targets, target_labels, inputs, labels = {}, [], {}, [], {}
    for key, row in by_id.items():
        decision = target_decision(row, registry[key], target_resolver)
        old = previous.get(key, {})
        if old.get("eligible") is False or (old.get("audit") or {}).get("eligible") is False:
            decision["eligible"] = False
            decision["reasons"] = sorted(set(decision["reasons"] + ["prior_rejection_requires_review"]))
        parsed = input_resolver(row)
        parent = parent_reference(row, parsed, registry) if valid_official_label(row.get("label")) else {
            "status": "unresolved", "reference_known": False,
            "reasons": ["missing_or_invalid_official_label"],
        }
        if parent["reference_known"]:
            source = {**registry[parent["producer_id"]], "consumed_path": parent["consumed_path"]}
            mutation = mutation_resolver(source, row, rows, registry)
            parent["mutable_source_audit"] = mutation
            if mutation["status"] != "clear":
                parent = {
                    "status": "unresolved", "reference_known": False,
                    "reasons": ["possible_checkpoint_mutation_or_reuse"],
                    "producer_id": parent["producer_id"], "mutable_source_audit": mutation,
                }
        decision["checkpoint_inputs"] = parsed
        decision["parent_reference"] = parent
        decision["one_step_eligible"] = decision["eligible"] and parent["reference_known"]
        decisions[key] = decision
        if not decision["eligible"]:
            continue
        example = project_example(row)
        targets.append(example)
        target_labels[key] = copy.deepcopy(row["label"])
        if not decision["one_step_eligible"]:
            continue
        example = copy.deepcopy(example)
        example["parent"] = {
            k: copy.deepcopy(parent[k]) for k in (
                "producer_id", "consumed_path", "accuracy", "evaluation_n", "path_kind",
                "reference_known", "official_grade_availability",
            )
        }
        inputs.append(example)
        labels[key] = {
            **copy.deepcopy(row["label"]),
            "delta_accuracy": row["label"]["accuracy"] - parent["accuracy"],
        }
    # Only pre-proposal, separately screened ancestor recipes become history.
    # A dirty parent recipe does not invalidate its independently bound score.
    for example in inputs:
        history, seen = [], {example["example_id"]}
        current = decisions[example["example_id"]]["parent_reference"]
        while current.get("reference_known"):
            parent_id = current["producer_id"]
            if parent_id in seen:
                raise ValueError("Cyclic checkpoint lineage")
            seen.add(parent_id)
            row, decision = by_id[parent_id], decisions[parent_id]
            history.append({
                "example_id": parent_id,
                "recipe_status": "screened" if decision["eligible"] else "quarantined",
                "model_input": project_example(row)["model_input"] if decision["eligible"] else None,
            })
            current = decision["parent_reference"]
        example["history"] = list(reversed(history))
        example["history_complete_to_base"] = (
            current.get("reasons") == ["base_official_protocol_score_unverified"]
            and all(h["recipe_status"] == "screened" for h in history)
        )
    return {
        "registry": registry, "decisions": decisions, "target_inputs": targets,
        "target_labels": target_labels, "one_step_inputs": inputs, "one_step_labels": labels,
    }


def freeze(previous=PREVIOUS_DATA, output=OUTPUT):
    from tools.outcome_prediction.wm_checkpoint_inputs import resolve_inputs
    from tools.outcome_prediction.wm_checkpoint_mutations import audit_source_use
    from tools.outcome_prediction.wm_checkpoint_paths import resolve_target_binding

    previous, output = Path(previous), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new versioned output directory")
    for relative, expected in read(previous / "manifest.json").items():
        if sha(previous / relative) != expected:
            raise ValueError("Changed input dataset: " + relative)
    rows = load_jsonl(previous / "private/inventory.jsonl")
    metadata = read(previous / "policy.json")["source_metadata"]
    raw_roots = {r["provenance"]["raw_root"] for r in rows}
    if len(raw_roots) != 1:
        raise ValueError("Expected one pinned raw root")
    raw_root = Path(next(iter(raw_roots)))
    trusted_raw = {}
    for source in metadata["files"]:
        path = raw_root / source["path"]
        if not path.resolve().is_relative_to(raw_root.resolve()):
            raise ValueError("Invalid raw source path")
        if sha(path) != source["sha256"]:
            raise ValueError("Changed pinned raw file: " + str(path))
        trusted_raw[str(path)] = source["sha256"]
    split = read(previous / "split.json")
    prior_source = metadata["previous_inventory"]
    if sha(prior_source["path"]) != prior_source["sha256"]:
        raise ValueError("Changed prior source review inventory")
    bundle = build(
        rows, split, resolve_target_binding, resolve_inputs,
        previous_rows=load_jsonl(prior_source["path"]), mutation_resolver=audit_source_use,
    )
    # Ensure the source snapshot also stayed unchanged while deriving new evidence.
    for path, expected in trusted_raw.items():
        if sha(path) != expected:
            raise ValueError("Raw source changed during cleanup: " + path)
    decisions = bundle["decisions"]
    summary = {}
    for benchmark in sorted({r["benchmark"] for r in rows}):
        part = [r for r in rows if r["benchmark"] == benchmark]
        kept = [r for r in part if decisions[r["example_id"]]["eligible"]]
        strict = [r for r in part if decisions[r["example_id"]]["one_step_eligible"]]
        summary[benchmark] = {
            "raw_cards": len(part),
            "official_labels": sum(valid_official_label(r.get("label")) for r in part),
            "screened_targets": len(kept),
            "screened_train": sum(split["cell_partition"][r["cell_id"]] == "train" for r in kept),
            "screened_test": sum(split["cell_partition"][r["cell_id"]] == "test" for r in kept),
            "labeled_quarantine": sum(valid_official_label(r.get("label")) for r in part) - len(kept),
            "unlabeled_quarantine": sum(not valid_official_label(r.get("label")) for r in part),
            "one_step_targets": len(strict),
            "one_step_train": sum(split["cell_partition"][r["cell_id"]] == "train" for r in strict),
            "one_step_test": sum(split["cell_partition"][r["cell_id"]] == "test" for r in strict),
            "one_step_sessions": len({r["cell_id"] for r in strict}),
            "one_step_train_sessions": len({r["cell_id"] for r in strict if split["cell_partition"][r["cell_id"]] == "train"}),
            "one_step_test_sessions": len({r["cell_id"] for r in strict if split["cell_partition"][r["cell_id"]] == "test"}),
            "screened_without_parent_reasons": dict(Counter(
                reason for r in kept for reason in decisions[r["example_id"]]["parent_reference"]["reasons"]
            )),
            "target_quarantine_reasons": dict(Counter(
                reason for r in part for reason in decisions[r["example_id"]]["reasons"]
            )),
        }
    policy = {
        "version": 1,
        "no_predictors_fitted": True,
        "no_score_imputation": True,
        "valid_zero_labels_retained": True,
        "score_magnitudes_not_used_for_selection": True,
        "split": "unchanged whole-session partitions from clean v2, assigned before filtering",
        "target": "first prospective recipe -> first archived checkpoint official accuracy",
        "parent": "unique exact same-session earlier archive/source path with a valid official score",
        "parent_recipe_not_required_for_parent_score": True,
        "known_modified_archives": sorted(ARCHIVE_REVIEW_HOLDS),
        "mutable_source_reuse": "bounded explicit mutation audit; unresolved changes quarantined",
        "base_scores": "not imputed; unverifiable protocol-specific baselines excluded from delta cohort",
        "multiple_inputs": "not reduced to a fabricated scalar parent",
        "model_input_contract": "Use fixed numeric whitelist extractors only; code is not unrestricted text-purity certified.",
        "limitations": [
            "Archive/source path and recorder-event correlation, not checkpoint byte certification.",
            "No checkpoint manifests or weight bytes are present in the pinned export.",
            "First proposal is a decision-time proxy, not a certified process launch time.",
            "Declared recipe equality and static save-path proofs do not certify the executed program or data bytes.",
            "Official parent grades were collected retrospectively; deployment assumes that score is supplied.",
            "AIME labels are single 30-question passes; sampling/decoding noise is not data corruption.",
            "History can be incomplete; quarantined ancestor recipes are withheld, not synthesized.",
            "Earlier test sessions have already been explored; this is a fixed development holdout, not a fresh confirmatory test.",
        ],
    }
    output.mkdir(parents=True)
    (output / "private").mkdir()
    for key in ("registry", "decisions"):
        write(output / "private" / (key + ".json"), bundle[key])
    for key in ("target_inputs", "target_labels", "one_step_inputs", "one_step_labels"):
        write(output / (key + ".json"), bundle[key])
    write(output / "split.json", {"cell_partition": split["cell_partition"]})
    write(output / "summary.json", summary)
    write(output / "policy.json", policy)
    sources = {
        str(previous / name): sha(previous / name)
        for name in ("private/inventory.jsonl", "split.json", "policy.json", "manifest.json")
    }
    for name in (
        "wm_one_step_data.py", "wm_checkpoint_inputs.py", "wm_checkpoint_paths.py",
        "wm_checkpoint_mutations.py", "rpm_code_provenance.py",
        "wm_clean_refresh.py", "wm_code_benchmark.py", "wm_code_features.py",
        "wm_grade_inventory.py",
    ):
        path = Path(__file__).parent / name
        sources[str(path)] = sha(path)
    for path in (Path("awm/wm/record.py"), raw_root / "README.md"):
        sources[str(path)] = sha(path)
    for name in ("base_scores_v1.json", "archive_integrity_v1.json"):
        path = Path("data/analysis/wm_one_step_audit") / name
        sources[str(path)] = sha(path)
    sources[prior_source["path"]] = prior_source["sha256"]
    manifest = {
        "sources": sources,
        "trusted_raw_file_count": len(trusted_raw),
        "trusted_raw_metadata_source": str(previous / "policy.json"),
        "files": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*.json"))},
    }
    write(output / "manifest.json", manifest)
    load_partition(output, "train")
    load_partition(output, "test")
    return summary


def load_partition(directory, partition, *, cohort="one_step"):
    """Fail closed: no unlabeled targets, missing parent, or modified frozen data."""
    if partition not in {"train", "test"} or cohort not in {"one_step", "target"}:
        raise ValueError("Invalid cohort or partition")
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    if not REQUIRED_FILES.issubset(manifest.get("files", {})):
        raise ValueError("Incomplete frozen manifest")
    if not manifest.get("sources"):
        raise ValueError("Missing source provenance")
    for source, expected in manifest["sources"].items():
        if sha(source) != expected:
            raise ValueError("Changed frozen source: " + source)
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Invalid manifest path")
        if sha(path) != expected:
            raise ValueError("Changed frozen data: " + str(path))
    examples = read(directory / (cohort + "_inputs.json"))
    labels = read(directory / (cohort + "_labels.json"))
    decisions = read(directory / "private/decisions.json")
    registry = read(directory / "private/registry.json")
    split = read(directory / "split.json")["cell_partition"]
    ids = [e["example_id"] for e in examples]
    if len(set(ids)) != len(ids) or set(ids) != set(labels):
        raise ValueError("Duplicate/misaligned feature/label identities")
    for example in examples:
        key = example["example_id"]
        decision = decisions[key]
        if (
            not decision["eligible"] or not valid_official_label(labels[key])
            or registry[key]["status"] != "official_archive_correlated"
            or labels[key]["accuracy"] != registry[key]["accuracy"]
            or registry[key]["cell_id"] != example["cell_id"]
            or registry[key]["benchmark"] != example["benchmark"]
            or split.get(example["cell_id"]) not in {"train", "test"}
        ):
            raise ValueError("Unlabeled or quarantined supervised row")
        if cohort == "one_step":
            parent = example["parent"]
            expected = registry[parent["producer_id"]]
            bound = decision.get("parent_reference") or {}
            if (
                not decision["one_step_eligible"] or parent["reference_known"] is not True
                or not bound.get("reference_known")
                or bound.get("status") != "official_archive_correlated"
                or any(parent.get(k) != bound.get(k) for k in (
                    "producer_id", "accuracy", "evaluation_n", "consumed_path",
                    "path_kind", "reference_known", "official_grade_availability",
                ))
                or expected["status"] != "official_archive_correlated"
                or expected["cell_id"] != example["cell_id"]
                or expected["benchmark"] != example["benchmark"]
                or expected["evaluation_n"] != labels[key]["evaluation_n"]
                or parent.get("evaluation_n") != expected["evaluation_n"]
                or parent.get("consumed_path") not in {expected["source_path"], expected["archive_path"]}
                or parent.get("path_kind") != (
                    "immutable_archive" if parent.get("consumed_path") == expected["archive_path"] else "archived_source"
                )
                or parent["producer_id"] == key
                or parent["accuracy"] != expected["accuracy"]
                or labels[key]["delta_accuracy"] != labels[key]["accuracy"] - parent["accuracy"]
            ):
                raise ValueError("Invalid observed-parent delta row")
    selected = [e for e in examples if split[e["cell_id"]] == partition]
    return selected, {e["example_id"]: labels[e["example_id"]] for e in selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=PREVIOUS_DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.previous, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
