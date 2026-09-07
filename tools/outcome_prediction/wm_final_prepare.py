"""Private complete-recipe drafts, not a semantic or execution certificate.

Assembly positively projects identity/task/plan/code and reconstruction metadata;
it never reads labels, eligibility, prior observations, or final snapshots. Every
inventory card remains represented, including unresolved cards and frozen runs
with zero approved targets. Scanner clearance never supplies whole-step review.

Data-only dependencies isolate a declared builder operation, not the producer's
training. Such isolation requires an additional external, content-bound review.
First-submission code evidence is a recorded version, not proof of actual launch
bytes or coverage of imports/data/configuration read indirectly by that code.
Honest unavailable code is represented explicitly and disclosed as a warning,
not mistaken for a complete-source requirement or fabricated from later files.

Only the separate ``join_train_labels`` function reads final target values, after
filtering each row by frozen TRAIN identity. Nothing in this module fits a model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from tools.outcome_prediction import wm_compile as compiler
from tools.outcome_prediction import wm_lineage as lineage
from tools.outcome_prediction import wm_recipe_input as recipe_input

SCHEMA = "wm-final-recipe-drafts-v1"
HASH = re.compile(r"[0-9a-f]{64}")
KINDS = {"weights", "generated_data", "configuration"}
REVIEW_FIELDS = {
    "payload_sha256",
    "reviewer",
    "evidence_sha256",
    "outcome_free",
    "executable_recipe_preserved",
}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _project(row):
    """Never copy a complete row/model_input/audit containing historical values."""
    source = row["model_input"]
    result = {
        key: row.get(key)
        for key in (
            "example_id",
            "cell_id",
            "card_id",
            "first_submitted_at",
        )
    }
    result["model_input"] = {
        key: copy.deepcopy(source.get(key, {} if key != "code" else []))
        for key in ("task", "plan", "code")
    }
    entries = _mapping(row.get("audit")).get("code_provenance", [])
    result["code_provenance"] = []
    for entry in entries:
        evidence = _mapping(entry.get("evidence"))
        result["code_provenance"].append(
            {
                **{
                    key: entry.get(key)
                    for key in (
                        "role",
                        "script_path",
                        "status",
                        "sha256",
                        "cutoff",
                        "trace_sha256",
                    )
                },
                "evidence_at": evidence.get("at"),
                "has_blockers": bool(entry.get("blockers")),
            }
        )
    return result


def split_cells(split):
    values = []
    for key in ("train_cell_ids", "test_cell_ids"):
        items = split.get(key)
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item for item in items)
            or len(items) != len(set(items))
        ):
            raise ValueError("Frozen split requires nonempty, unique run IDs")
        values.append(set(items))
    train, test = values
    if train & test:
        raise ValueError("Frozen train/test runs overlap")
    return train, test


def _command(setup):
    command, launch = _mapping(setup.get("command")), _mapping(setup.get("launch"))
    return command or launch


def _role(plan):
    setup = _mapping(plan.get("setup"))
    family = str(_mapping(setup.get("method")).get("family", "")).lower()
    if re.search(r"merge|soup|averag", family):
        return "merge"
    if re.search(r"decod|config|packag", family):
        return "decoding"
    if re.search(r"eval|selection", family):
        return "evaluation"
    if family.replace("-", "_").replace(" ", "_") in {
        "sft",
        "distill",
        "distillation",
        "grpo",
        "ppo",
        "dpo",
        "rft",
        "rl",
        "training",
        "pretrain",
        "finetune",
        "full_finetune",
        "continued_pretraining",
    }:
        return "training"
    command = _command(setup)
    text = json.dumps(command, sort_keys=True).lower()
    for role, pattern in (
        ("merge", r"merge|soup|average_weights"),
        ("training", r"(?:train|finetune|distill)[a-z0-9_.-]*\.py"),
        ("evaluation", r"(?:eval|evaluate|sweep|select_checkpoint)[a-z0-9_.-]*\.py"),
        ("decoding", r"generation_config|decod"),
    ):
        if re.search(pattern, text):
            return role
    return "unknown"


def _issue(kind, step_id, **detail):
    return {"kind": kind, "step_id": step_id, **detail}


def _manual_source_hold_issues(audit, step_id, source_id):
    """Unresolved source reviews are independent of payload-hash approvals."""
    holds = audit.get("manual_source_holds", [])
    if not isinstance(holds, list):
        raise TypeError("Malformed compiled manual source holds")
    fields = ("path", "review_id", "source_sha256", "evidence_sha256")
    issues = []
    for hold in holds:
        if (
            not isinstance(hold, dict)
            or any(not isinstance(hold.get(key), str) or not hold[key].strip() for key in fields)
            or not hold["path"].startswith("/")
            or not HASH.fullmatch(hold["source_sha256"])
            or not HASH.fullmatch(hold["evidence_sha256"])
        ):
            raise ValueError("Malformed compiled manual source hold")
        issues.append(
            _issue(
                "compiler_manual_source_hold",
                step_id,
                source_example_id=source_id,
                **{key: hold[key] for key in fields},
            )
        )
    return issues


def _command_paths(command, cwd):
    """Literal scripts/configs only; shell is parsed, never executed."""
    segments, warnings = lineage._segments(command.get("argv", command.get("command")))
    paths = set()
    declared_script = command.get("script")
    if isinstance(declared_script, str):
        paths.add(lineage._artifact(declared_script, cwd))
    for segment in segments:
        for token in segment:
            if re.fullmatch(r"[^\s]+\.(?:py|sh)", token) and not token.startswith("--"):
                paths.add(lineage._artifact(token, cwd))
        for flag, values in lineage._flags(segment):
            if flag in lineage.CONFIG_FLAGS:
                for value in values:
                    if value.endswith((".json", ".yaml", ".yml")):
                        paths.add(lineage._artifact(value, cwd))
    return paths, warnings


def _coverage(row, draft, node):
    step_id = row["example_id"]
    issues, records, coverage_warnings = [], [], []
    source_codes = row["model_input"]["code"]
    setup = _mapping(draft["plan"].get("setup"))
    command = _command(setup)
    cwd = command.get("cwd") or node.get("cwd")
    if (
        _mapping(setup.get("command"))
        and _mapping(setup.get("launch"))
        and setup["command"] != setup["launch"]
    ):
        issues.append(_issue("conflicting_command_and_launch", step_id))
    required, warnings = _command_paths(command, cwd)
    for data in setup.get("data", []):
        if isinstance(data, dict) and data.get("build_command"):
            paths, extra = _command_paths(
                {
                    "argv": data["build_command"],
                    "script": data.get("built_by"),
                },
                cwd,
            )
            required.update(paths)
            warnings.extend(extra)
    if setup.get("resume_argv"):
        paths, extra = _command_paths({"argv": setup["resume_argv"]}, cwd)
        required.update(paths)
        warnings.extend(extra)
    for warning in sorted(set(warnings)):
        coverage_warnings.append(_issue("command_not_statically_covered", step_id, reason=warning))
    by_path = {}
    for code in draft["code"]:
        source_status = code.get("status")
        if source_status in {"not_requested", "blocked"} and code.get("content") is None:
            code["status"] = "unavailable"
        path = lineage._artifact(code.get("script_path"), cwd)
        if path:
            by_path.setdefault(path, []).append(code)
        if code.get("status") == "not_declared" and code.get("script_path") is None:
            continue
        if code.get("status") != "reconstructed":
            coverage_warnings.append(
                _issue(
                    "code_not_reconstructed",
                    step_id,
                    script_path=code.get("script_path"),
                    source_availability_status=source_status,
                )
            )
            continue
        matches = [
            c
            for c in source_codes
            if all(
                c.get(k) == code.get(k)
                for k in (
                    "role",
                    "script_path",
                    "status",
                )
            )
        ]
        evidence = [
            p
            for p in row["code_provenance"]
            if all(p.get(k) == code.get(k) for k in ("role", "script_path", "status"))
        ]
        reasons = []
        if not isinstance(code.get("content"), str):
            reasons.append("reconstructed_code_missing_content")
        if len(matches) != 1 or len(evidence) != 1:
            reasons.append("missing_or_ambiguous_first_submit_provenance")
        else:
            original, proof = matches[0], evidence[0]
            text = original.get("content")
            if not isinstance(text, str) or proof.get("sha256") != compiler.text_digest(text):
                reasons.append("source_code_hash_mismatch")
            cutoff = lineage._time(row["first_submitted_at"])
            at = lineage._time(proof.get("evidence_at"))
            if (
                not cutoff
                or not at
                or at >= cutoff
                or proof.get("cutoff") != row["first_submitted_at"]
            ):
                reasons.append("code_not_proven_before_first_submission")
            if not HASH.fullmatch(proof.get("trace_sha256") or "") or proof["has_blockers"]:
                reasons.append("incomplete_reconstruction_evidence")
            records.append(
                {
                    "role": code.get("role"),
                    "script_path": code.get("script_path"),
                    "source_content_sha256": proof.get("sha256"),
                    "draft_content_sha256": compiler.text_digest(code["content"])
                    if isinstance(code.get("content"), str)
                    else None,
                    "cutoff": proof.get("cutoff"),
                    "evidence_at": proof.get("evidence_at"),
                    "trace_sha256": proof.get("trace_sha256"),
                }
            )
        for reason in reasons:
            issues.append(_issue(reason, step_id, script_path=code.get("script_path")))
    for path in sorted(required):
        found = by_path.get(path, [])
        if not found:
            draft["code"].append(
                {
                    "role": "declared_dependency",
                    "script_path": path,
                    "status": "unavailable",
                    "content": None,
                }
            )
        if not found or not any(code.get("status") == "reconstructed" for code in found):
            coverage_warnings.append(
                _issue(
                    "declared_script_or_config_not_covered",
                    step_id,
                    artifact=path,
                    represented_as="unavailable",
                )
            )
    return issues, {
        "required_paths": sorted(required),
        "recorded_code": records,
        "transitive_imports_and_data_verified": False,
        "actual_launch_verified": False,
        "coverage_warnings": coverage_warnings,
    }


def _isolate_builder(producer, draft, node, artifact):
    """Return a proposed operation, never assert its isolation is certified."""
    indices = set()
    for output in node["outputs"]:
        if output["kind"] == "generated_data" and output["artifact"] == artifact:
            match = re.match(r"setup\.data\[(\d+)\]", output["evidence"])
            if match:
                indices.add(int(match[1]))
    if len(indices) != 1:
        return None
    index = indices.pop()
    data = _mapping(draft["plan"].get("setup")).get("data", [])
    if index >= len(data) or not isinstance(data[index], dict):
        return None
    item = data[index]
    if not item.get("build_command") or not isinstance(item.get("built_by"), str):
        return None
    segments, warnings = lineage._segments(item["build_command"])
    if not segments or warnings:
        return None
    path = lineage._artifact(item["built_by"], node.get("cwd"))
    codes = [
        copy.deepcopy(c)
        for c in draft["code"]
        if lineage._artifact(c.get("script_path"), node.get("cwd")) == path
        and c.get("role") == f"data_builder_{index}"
    ]
    if len(codes) != 1 or codes[0].get("status") != "reconstructed":
        return None
    operation = {
        "step_id": f"{producer}#data-{index}",
        "role": "data_generation",
        "parents": [],
        "plan": {
            "setup": {
                "command": {
                    "cwd": node.get("cwd"),
                    "argv": copy.deepcopy(item["build_command"]),
                    "script": item["built_by"],
                },
                "data": [copy.deepcopy(item)],
            }
        },
        "code": codes,
    }
    prefix = f"setup.data[{index}]."
    dependencies = [
        (i, edge)
        for i, edge in enumerate(node["parents"])
        if any(field.startswith(prefix) for field in edge["evidence_fields"])
    ]
    return operation, dependencies, index


def _self_generated_provenance_issues(draft, node, *, only_data_index=None):
    """Supplement the frozen graph; never invent producers or inspect outcomes.

    The v2 lineage parser does not recognize every generator/merge input flag.
    Explicit self-generation declarations and --rft inputs therefore need a
    corresponding declared generator/data dependency, even when v2 says clear.
    This narrow diagnostic is not a replacement for whole-recipe review.
    """
    issues = []
    setup = _mapping(draft["plan"].get("setup"))
    for index, data in enumerate(setup.get("data") or []):
        if only_data_index is not None and index != only_data_index:
            continue
        if not isinstance(data, dict):
            continue
        prefix = f"setup.data[{index}]."
        source = data.get("source")
        declares_self = isinstance(source, str) and bool(
            re.search(
                r"synthetic\s*:\s*self\b|\bself[- ](?:generated|sampled|sampling)\b|\bon[- ]policy\s+(?:rft|rollouts?)\b",
                source,
                re.IGNORECASE,
            )
        )
        segments, _ = lineage._segments(data.get("build_command"))
        inputs = {
            lineage._artifact(value, node.get("cwd"))
            for segment in segments
            for flag, values in lineage._flags(segment)
            if flag in {"rft", "rft_data", "rollouts", "generations", "samples_file"}
            for value in lineage._values(values)
        }
        edges = [
            edge
            for edge in node["parents"]
            if any(field.startswith(prefix) for field in edge["evidence_fields"])
        ]
        resolved = [
            edge
            for edge in edges
            if edge["resolution_status"]
            in {
                "time_qualified_declared_artifact",
                "declared_base_model",
            }
        ]
        for artifact in sorted(inputs):
            if not any(
                edge["artifact"] == artifact and edge["kind"] == "generated_data"
                for edge in resolved
            ):
                issues.append(
                    _issue(
                        "self_generated_input_provenance_unresolved",
                        node["example_id"],
                        path=f"/plan/setup/data/{index}/build_command",
                        artifact=artifact,
                        source_sha256=compiler.digest(data.get("build_command")),
                        reason="Explicit rollout/RFT input has no resolved exact data-artifact edge in the frozen graph; do not infer its producer from a filename.",
                    )
                )
        if (
            declares_self
            and not inputs
            and not any(
                edge["kind"] == "generated_data"
                and any(
                    field.startswith(prefix) and field.endswith((".generator", ".input", ".path"))
                    for field in edge["evidence_fields"]
                )
                for edge in resolved
            )
        ):
            issues.append(
                _issue(
                    "self_generation_operation_provenance_unresolved",
                    node["example_id"],
                    path=f"/plan/setup/data/{index}/source",
                    source_sha256=compiler.digest(source),
                    reason="Explicit self-generated data has no resolved generator or exact reusable-data dependency; an unrelated training parent is insufficient.",
                )
            )
    return issues


def _review(step, reviews):
    """Lookup by exact step-content hash, not an unconstrained source-card ID."""
    review = reviews.get(recipe_input.digest(step))
    if review is None:
        return None
    if not isinstance(review, dict) or set(review) != REVIEW_FIELDS:
        raise ValueError("Malformed whole-step review")
    # Review identity is checked here; the final DAG is checked by its caller.
    if review["payload_sha256"] != recipe_input.digest(step) or not (
        review["outcome_free"] is True
        and review["executable_recipe_preserved"] is True
        and isinstance(review["reviewer"], str)
        and review["reviewer"].strip()
        and HASH.fullmatch(review["evidence_sha256"] or "")
    ):
        raise ValueError("Unapproved or stale whole-step review")
    return copy.deepcopy(review)


def assemble(
    rows, graph, split, *, compiled_steps=None, whole_step_reviews=None, operation_reviews=None
):
    """Assemble every card; no label-based target/ancestor selection is performed.

    Reviewed compiler results bind ``audit.source_recipe_projection_sha256`` to
    the original task/plan/code projection. Operation-review values require the
    operation's payload hash, source projection hash, dependency edge hash, named
    reviewer/evidence hash, and ``operation_isolation_verified: true``. They do
    not replace the separate whole-step semantic review.
    """
    projected = [_project(row) for row in rows]
    by_id = {r["example_id"]: r for r in projected}
    if len(by_id) != len(projected):
        raise ValueError("Duplicate inventory cards")
    train, test = split_cells(split)
    if {r["cell_id"] for r in projected} - train - test:
        raise ValueError("Inventory contains runs outside the frozen split")
    expected = lineage.build_graph(projected)
    if graph != expected:
        raise ValueError(
            "Lineage graph is stale or its edges/closure differ from source declarations"
        )
    if compiled_steps is None:
        compiled_steps, _ = compiler.compile_steps(projected)
    compiled = {r["step_id"]: r for r in compiled_steps}
    if len(compiled) != len(compiled_steps) or set(compiled) != set(by_id):
        raise ValueError("Compiled drafts must cover every inventory card exactly once")
    for key, result in compiled.items():
        source_hash = compiler.digest(by_id[key]["model_input"])
        audit = result["audit"]
        bound = audit.get("source_recipe_projection_sha256", audit.get("source_input_sha256"))
        if bound != source_hash or audit["draft_input_sha256"] != compiler.digest(
            result["draft_input"]
        ):
            raise ValueError("Stale source projection or compiler draft hash")
        if result["draft_input"]["task"] != by_id[key]["model_input"]["task"]:
            raise ValueError("Reviewed compiler draft changed task identity")
    whole_step_reviews, operation_reviews = whole_step_reviews or {}, operation_reviews or {}
    output, review_queue = [], {}
    nodes = graph["nodes"]
    for target in sorted(by_id):
        steps, sources, provenance, issues, visiting = {}, {}, {}, [], set()
        operation_ids = set()

        def expand(
            source_id,
            artifact=None,
            *,
            steps=steps,
            sources=sources,
            provenance=provenance,
            issues=issues,
            visiting=visiting,
            operation_ids=operation_ids,
            target=target,
        ):
            row, node, result = by_id[source_id], nodes[source_id], compiled[source_id]
            draft = result["draft_input"]
            data_index = None
            dependencies = list(enumerate(node["parents"]))
            if artifact is not None:
                isolated = _isolate_builder(source_id, draft, node, artifact)
                if isolated is None:
                    issues.append(
                        _issue(
                            "data_builder_operation_not_isolatable", source_id, artifact=artifact
                        )
                    )
                    return None
                step, dependencies, data_index = isolated
                operation_ids.add(step["step_id"])
            else:
                step = {
                    "step_id": source_id,
                    "role": _role(draft["plan"]),
                    "parents": [],
                    "plan": copy.deepcopy(draft["plan"]),
                    "code": copy.deepcopy(draft["code"]),
                }
            step_id = step["step_id"]
            if step_id in visiting:
                raise ValueError("Cycle in operation-expanded recipe")
            if step_id in steps:
                return step_id
            visiting.add(step_id)
            if result["status"] == "rejected":
                issues.append(_issue("compiler_rejected_source", step_id))
            # Do not resolve/narrow holds using scanner state, status, a content
            # approval, or builder isolation. Only explicit source review may do so.
            issues.extend(_manual_source_hold_issues(result["audit"], step_id, source_id))
            if any(
                draft["task"].get(key) != by_id[target]["model_input"]["task"].get(key)
                for key in ("benchmark", "base_model")
            ):
                issues.append(_issue("ancestor_task_identity_mismatch", step_id))
            # Source edits may not quietly change typed dependency declarations.
            issues.extend(
                {**issue, "step_id": step_id}
                for issue in _self_generated_provenance_issues(
                    draft, node, only_data_index=data_index
                )
            )
            if data_index is None:
                changed_row = {**row, "model_input": draft}
                if lineage._project(changed_row) != lineage._project(row):
                    issues.append(_issue("review_changed_lineage_declarations", step_id))
                for code in row["model_input"]["code"]:
                    if code.get("status") == "reconstructed" and not any(
                        all(other.get(k) == code.get(k) for k in ("role", "script_path", "status"))
                        for other in draft["code"]
                    ):
                        issues.append(
                            _issue(
                                "reconstructed_source_identity_changed",
                                step_id,
                                script_path=code.get("script_path"),
                            )
                        )
            edge_records = []
            for index, edge in dependencies:
                edge_records.append({"edge_index": index, **copy.deepcopy(edge)})
                status = edge["resolution_status"]
                if status in {"declared_base_model", "internal_declared_operation"}:
                    continue
                if status != "time_qualified_declared_artifact" or not edge.get("artifact"):
                    issues.append(
                        _issue(
                            "unresolved_dependency",
                            step_id,
                            edge_index=index,
                            status=status,
                            artifact=edge.get("artifact"),
                        )
                    )
                    continue
                parent = edge["producer_id"]
                if parent not in by_id or by_id[parent]["cell_id"] != row["cell_id"]:
                    raise ValueError("Missing or cross-run producer")
                scope = edge["dependency_scope"]
                if scope not in {
                    "data_builder_only",
                    "requires_producer_weights",
                    "configuration_only",
                }:
                    issues.append(_issue("unresolved_dependency_scope", step_id, edge_index=index))
                    continue
                parent_step = expand(
                    parent, edge["artifact"] if scope == "data_builder_only" else None
                )
                if parent_step is None:
                    continue
                kind = "weights" if scope == "requires_producer_weights" else edge["kind"]
                if kind not in KINDS:
                    raise ValueError("Unknown typed dependency")
                link = {"step_id": parent_step, "kind": kind, "artifact": edge["artifact"]}
                if link not in step["parents"]:
                    step["parents"].append(link)
            visiting.remove(step_id)
            coverage_issues, coverage = _coverage(
                row, {"plan": step["plan"], "code": step["code"]}, node
            )
            issues.extend({**i, "step_id": step_id} for i in coverage_issues)
            if data_index is not None:
                isolation_hash = recipe_input.digest(step)
                proof = operation_reviews.get(isolation_hash)
                required = {
                    "payload_sha256": isolation_hash,
                    "source_recipe_projection_sha256": compiler.digest(row["model_input"]),
                    "dependency_edges_sha256": compiler.digest(edge_records),
                    "operation_isolation_verified": True,
                }
                if proof is None:
                    issues.append(
                        _issue("operation_isolation_review_required", step_id, **required)
                    )
                elif (
                    not isinstance(proof, dict)
                    or any(proof.get(k) != v for k, v in required.items())
                    or not (
                        isinstance(proof.get("reviewer"), str)
                        and proof["reviewer"].strip()
                        and HASH.fullmatch(proof.get("evidence_sha256") or "")
                    )
                ):
                    raise ValueError("Invalid operation-isolation review")
            steps[step_id] = step
            sources[step_id] = source_id
            provenance[step_id] = {
                "source_example_id": source_id,
                "operation_data_index": data_index,
                "first_submitted_at": row["first_submitted_at"],
                "source_recipe_projection_sha256": compiler.digest(row["model_input"]),
                "compiled_draft_sha256": result["audit"]["draft_input_sha256"],
                "declared_dependencies": edge_records,
                "code_coverage": coverage,
                "unresolved_fragment_finding_ids": [
                    f["finding_id"]
                    for f in result["audit"]["unresolved_findings"]
                    if data_index is None
                    or f["path"].startswith(f"/plan/setup/data/{data_index}/")
                    or f["path"].startswith("/code/")
                ],
            }
            return step_id

        final = expand(target)
        payload = {
            "schema": recipe_input.SCHEMA,
            "task": copy.deepcopy(by_id[target]["model_input"]["task"]),
            "recipe": {"steps": list(steps.values()), "final_step_id": final},
        }
        structurally_valid = True
        try:
            recipe_input.validate_full_recipe(payload)
        except (ValueError, TypeError) as error:
            structurally_valid = False
            issues.append(_issue("invalid_recipe_structure", target, reason=str(error)))
        reviews = {}
        for step in steps.values():
            step_hash = recipe_input.digest(step)
            review = _review(step, whole_step_reviews)
            if review is not None:
                reviews[step["step_id"]] = review
            else:
                issues.append(
                    _issue(
                        "whole_step_content_review_required",
                        step["step_id"],
                        payload_sha256=step_hash,
                    )
                )
                review_queue.setdefault(
                    step_hash,
                    {
                        "payload_sha256": step_hash,
                        "step": copy.deepcopy(step),
                        "source_example_id": sources[step["step_id"]],
                        "operation_isolation_required": step["step_id"] in operation_ids,
                        "semantic_review_focus": [
                            "Distinguish prospective numeric generation budgets from realized correctness-filtered yields/coverage, including setup.data.n_examples and selection text.",
                            "Determine which selection/promotion operations generated the archive-scored checkpoint. Do not substitute a local protocol sample count for task.evaluation_n or remove a screen that selected exported weights.",
                        ],
                        "target_ids": [],
                    },
                )["target_ids"].append(target)
        approved = not issues
        if approved:
            recipe_input.build_reviewed_input(
                task=payload["task"],
                steps=payload["recipe"]["steps"],
                final_step_id=final,
                reviews=reviews,
            )
        content_review = {
            "status": "approved" if approved else "needs_review",
            "payload_sha256": recipe_input.digest(payload),
            "step_reviews": reviews,
        }
        output.append(
            {
                "example_id": target,
                "cell_id": by_id[target]["cell_id"],
                "partition": "train" if by_id[target]["cell_id"] in train else "test",
                "status": "approved" if approved else "draft_needs_review",
                "model_input": payload,
                "content_review": content_review,
                "step_sources": sources,
                "step_provenance": provenance,
                "audit": {
                    "structurally_valid": structurally_valid,
                    "issues": issues,
                    "label_values_used": False,
                    "execution_certified": False,
                    "target_definition": "immediate_per_card_official_checkpoint_accuracy",
                    "target_evaluation_n": payload["task"].get("evaluation_n"),
                    "coverage_warnings": [
                        warning
                        for p in provenance.values()
                        for warning in p["code_coverage"]["coverage_warnings"]
                    ],
                    "card_level_closure": nodes[target]["topological_closure"],
                    "retained_source_ids": sorted(set(sources.values())),
                    "operation_expanded_closure": list(steps),
                },
            }
        )
    coverage = []
    for cell in sorted(train | test):
        subset = [r for r in output if r["cell_id"] == cell]
        coverage.append(
            {
                "cell_id": cell,
                "partition": "train" if cell in train else "test",
                "inventory_targets": len(subset),
                "structurally_valid_targets": sum(r["audit"]["structurally_valid"] for r in subset),
                "approved_targets": sum(r["status"] == "approved" for r in subset),
            }
        )
    return {
        "schema": SCHEMA,
        "drafts": output,
        "review_queue": list(review_queue.values()),
        "audit": {
            "label_values_used": False,
            "semantic_certificate": False,
            "inventory_cards": len(output),
            "frozen_train_runs": len(train),
            "frozen_test_runs": len(test),
            "split_sha256": compiler.digest(split),
            "graph_sha256": compiler.digest(graph),
            "run_coverage": coverage,
            "runs_without_approved_targets": [
                r["cell_id"] for r in coverage if not r["approved_targets"]
            ],
            "issue_counts": dict(Counter(i["kind"] for r in output for i in r["audit"]["issues"])),
            "coverage_warning_counts": dict(
                Counter(
                    warning["kind"] for r in output for warning in r["audit"]["coverage_warnings"]
                )
            ),
            "limits": [
                "Private drafts may still contain implicit outcome language; never expose to prediction agents.",
                "All cards retained; no shorter/terminal target policy or eligibility selection inferred.",
                "Target is the immediate official checkpoint, not a subsequently promoted deployed champion; pre-official selection belongs in the recipe and post-official shipping rules require separate scope review.",
                "Typed declared artifacts and first-submission code are not execution certificates.",
                "Split membership is immutable; empty runs remain explicit gaps.",
            ],
        },
    }


def join_train_labels(prepared, rows, split):
    """Explicit final-label phase. Test rows are filtered BEFORE any other access.

    Missing/nonofficial final labels remain gaps; no ancestor label is inserted
    in model_input, and no heldout scores or features are inspected here.
    """
    train, test = split_cells(split)
    if prepared["audit"]["split_sha256"] != compiler.digest(split):
        raise ValueError("Frozen split changed after draft assembly")
    approved = {
        r["example_id"]: r
        for r in prepared["drafts"]
        if r["cell_id"] in train and r["status"] == "approved"
    }
    output, seen, gaps = [], set(), []
    for row in rows:
        cell = row["cell_id"]
        if cell in test:
            continue
        if cell not in train:
            raise ValueError("Label source contains unknown run")
        key = row["example_id"]
        if key in seen:
            raise ValueError("Duplicate TRAIN label source")
        seen.add(key)
        if key not in approved:
            continue
        selected = approved[key]
        if selected["cell_id"] != cell:
            raise ValueError("Final label identity mismatch")
        if selected["audit"]["issues"] or not selected["audit"]["structurally_valid"]:
            raise ValueError("Cannot join labels to unresolved drafts")
        payload = selected["model_input"]
        review = selected["content_review"]
        sources = selected["step_sources"]
        if (
            sources.get(payload["recipe"]["final_step_id"]) != key
            or set(sources) != {step["step_id"] for step in payload["recipe"]["steps"]}
            or any(not source.startswith(cell + "/") for source in sources.values())
        ):
            raise ValueError("Final target or ancestor source identity mismatch")
        if review["status"] != "approved" or review["payload_sha256"] != recipe_input.digest(
            payload
        ):
            raise ValueError("Stale approved payload")
        recipe_input.build_reviewed_input(
            task=payload["task"],
            steps=payload["recipe"]["steps"],
            final_step_id=payload["recipe"]["final_step_id"],
            reviews=review["step_reviews"],
        )
        label = row["label"]
        accuracy = label.get("accuracy")
        official = label.get("official_metric")
        if (
            type(accuracy) not in (int, float)
            or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1
            or not isinstance(official, dict)
            or type(official.get("accuracy")) not in (int, float)
            or official["accuracy"] != accuracy
        ):
            gaps.append({"example_id": key, "reason": "missing_valid_official_final_label"})
            continue
        output.append(
            {
                key: copy.deepcopy(selected[key])
                for key in (
                    "example_id",
                    "cell_id",
                    "model_input",
                    "content_review",
                    "step_sources",
                )
            }
            | {"label": {"accuracy": accuracy}}
        )
    gaps.extend(
        {"example_id": key, "reason": "missing_train_label_row"}
        for key in sorted(set(approved) - seen)
    )
    fitted = {r["cell_id"] for r in output}
    return output, {
        "phase": "explicit_train_final_label_join",
        "test_labels_read": False,
        "train_rows": len(output),
        "frozen_train_cell_ids": sorted(train),
        "represented_train_cell_ids": sorted(fitted),
        "empty_train_cell_ids": sorted(train - fitted),
        "label_gaps": gaps,
        "split_was_not_reassigned": True,
    }


def _write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_whole_step_reviews(paths):
    """Load explicit trusted decisions, checking their complete-read evidence.

    Hash integrity is not an independent semantic review. These PRIVATE evidence
    documents can contain removed observations and must never enter agent inputs.
    """
    reviews, provenance = {}, []
    for path in paths:
        raw = Path(path).read_bytes()
        document = json.loads(raw)
        if document.get("schema") not in {
            "wm-whole-step-review-batch-v1",
            "wm-whole-step-manual-review-v1",
        }:
            raise ValueError("Unknown whole-step review document schema")
        evidence = {}
        for wrapper in document.get("evidence_entries", []):
            entry = wrapper.get("evidence", wrapper)
            entry_hash = compiler.digest(entry)
            if "evidence" in wrapper and wrapper.get("evidence_sha256") != entry_hash:
                raise ValueError("Stale whole-step evidence hash")
            evidence[entry_hash] = entry
        mapping = document.get("reviews_map")
        if not isinstance(mapping, dict):
            raise TypeError("Whole-step review map is required")
        for key, review in mapping.items():
            if (
                not HASH.fullmatch(key)
                or not isinstance(review, dict)
                or set(review) != REVIEW_FIELDS
                or review.get("payload_sha256") != key
                or review.get("outcome_free") is not True
                or review.get("executable_recipe_preserved") is not True
                or not isinstance(review.get("reviewer"), str)
                or not review["reviewer"].strip()
            ):
                raise ValueError("Invalid whole-step approval")
            entry = evidence.get(review.get("evidence_sha256"))
            if entry is None or entry.get("payload_sha256") != key:
                raise ValueError("Missing or mismatched whole-step evidence")
            if document["schema"] == "wm-whole-step-review-batch-v1":
                approved = entry.get("decision") == "approved" and all(
                    entry.get(flag) is True
                    for flag in (
                        "full_step_read",
                        "full_plan_read",
                        "all_supplied_code_read",
                        "declared_parents_inspected",
                    )
                )
            else:
                approved = entry.get("status") == "approved" and all(
                    entry.get(flag) is True
                    for flag in ("complete_step_read", "complete_supplied_code_read")
                )
            if not approved or entry.get("findings"):
                raise ValueError("Pending or incompletely read step cannot be approved")
            if key in reviews and reviews[key] != review:
                raise ValueError("Conflicting duplicate whole-step approvals")
            reviews[key] = copy.deepcopy(review)
        provenance.append(
            {
                "path": str(path),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "document_sha256": compiler.digest(document),
                "approved_step_hashes": sorted(mapping),
            }
        )
    return reviews, provenance


def build_artifact(
    inventory_path,
    graph_path,
    split_path,
    output_dir,
    *,
    compiled_path=None,
    whole_step_review_paths=(),
):
    """Create a new private draft artifact. No labels are emitted or selected."""
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a NEW immutable draft directory")
    raw = Path(inventory_path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    graph_bytes, split_bytes = Path(graph_path).read_bytes(), Path(split_path).read_bytes()
    compiled_bytes = Path(compiled_path).read_bytes() if compiled_path else None
    graph, split = json.loads(graph_bytes), json.loads(split_bytes)
    compiled = json.loads(compiled_bytes) if compiled_bytes else None
    reviews, review_provenance = load_whole_step_reviews(whole_step_review_paths)
    result = assemble(rows, graph, split, compiled_steps=compiled, whole_step_reviews=reviews)
    current_hashes = {
        compiler.digest(step)
        for draft in result["drafts"]
        for step in draft["model_input"]["recipe"]["steps"]
    }
    if set(reviews) - current_hashes:
        raise ValueError("Whole-step approval does not match any current assembled step")
    output.mkdir(parents=True, mode=0o700)
    _write_json(output / "drafts.json", result["drafts"])
    _write_json(output / "review_queue.json", result["review_queue"])
    _write_json(output / "audit.json", result["audit"])
    _write_json(
        output / "provenance.json",
        {
            "schema": SCHEMA,
            "inventory_sha256": hashlib.sha256(raw).hexdigest(),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_module_sha256": {
                module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in (compiler, lineage, recipe_input)
            },
            "graph_file_sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "split_file_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "compiled_file_sha256": hashlib.sha256(compiled_bytes).hexdigest()
            if compiled_bytes
            else None,
            "whole_step_reviews": review_provenance,
            "result_sha256": compiler.digest(result),
            "label_values_used": False,
            "files_sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ("drafts.json", "review_queue.json", "audit.json")
            },
        },
    )
    for path in output.iterdir():
        path.chmod(0o600)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--compiled", type=Path)
    parser.add_argument("--whole-step-review", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_artifact(
        args.inventory,
        args.graph,
        args.split,
        args.output_dir,
        compiled_path=args.compiled,
        whole_step_review_paths=args.whole_step_review,
    )
    summary = {
        key: value
        for key, value in result["audit"].items()
        if key not in {"run_coverage", "runs_without_approved_targets", "limits"}
    }
    summary["runs_without_approved_target_count"] = len(
        result["audit"]["runs_without_approved_targets"]
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
