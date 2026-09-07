"""Private, hash-bound source-fragment review integration; never step approval.

Review artifacts are trusted reviewer decisions, not instructions discovered in
trajectory text. This module validates and applies only their exact deletions.
The in-memory integrator does not write files, read labels, run recovered code,
or grant an agent payload. The explicit CLI persists new private draft bundles.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import math
import os
import re
import tokenize
from collections import Counter
from itertools import pairwise
from pathlib import Path

from tools.outcome_prediction.wm_compile import (
    _code_findings,
    _scan_operations,
    compile_steps,
    digest,
    text_digest,
)

SCHEMA = "wm-source-review-integration-v1"
SCOPES = {
    "wm-code-fragment-review-v1": "code_fragments_only_not_whole_step",
    "wm-source-fragment-review-v1": "source_fragments_only_not_whole_step",
    "wm-plan-fragment-review-v1": "plan_operational_fragments_only_not_whole_step",
}
STATUSES = {"approve_findings", "redact_nonexecuting_text", "remove_field", "needs_review"}
HASH = re.compile(r"[0-9a-f]{64}")
CODE_PATH = re.compile(r"/code/(0|[1-9][0-9]*)/content")
MANUAL_SCHEMA = "wm-manual-source-field-review-v1"
MANUAL_SCOPE = "manual_source_fields_only_not_whole_step"
NUMERIC_SCHEMA = "wm-observed-numeric-metadata-review-v1"
NUMERIC_SCOPE = "observed_numeric_metadata_deletion_only_not_whole_step"
NUMERIC_PATH = re.compile(r"/plan/setup/(?:data/(0|[1-9][0-9]*)/n_examples|progress/total)")


def _hash(value, name):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"Invalid {name} SHA256")
    return value


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Nonempty {name} is required")
    return value


def _parts(path):
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("Invalid source occurrence path")
    if re.search(r"~(?![01])", path):
        raise ValueError("Invalid JSON pointer escape")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
    if parts[0] != "plan" and not CODE_PATH.fullmatch(path):
        raise ValueError("Only plan string fields and recovered code content may be reviewed")
    return parts


def _lookup(value, parts):
    for part in parts:
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", part):
                raise ValueError("Noncanonical list index in occurrence")
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise TypeError("Occurrence does not identify a field")
    return value


def load_review_documents(paths):
    """Read explicit review files, preserving raw-file and parsed-content hashes."""
    documents = []
    for path in paths:
        raw = Path(path).read_bytes()
        document = json.loads(raw)
        documents.append(
            {
                "review_document": document,
                "artifact_path": str(path),
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "document_sha256": digest(document),
            }
        )
    return documents


def review_drafts_digest(steps):
    """Bind the entire CURRENT draft stage, not original or assembled payloads.

    The value is wm_compile.digest({step_id: digest(draft_input), ...}). JSON
    mapping keys are canonicalized by digest, so input list order is irrelevant.
    Only positive draft inputs and their identities/hashes are consulted.
    """
    identities = {}
    for step in steps:
        identity = _nonempty(step["step_id"], "step_id")
        if identity in identities:
            raise ValueError("Ambiguous duplicate step IDs")
        current = digest(step["draft_input"])
        if current != step["audit"]["draft_input_sha256"]:
            raise ValueError("Stale compiler draft hash")
        identities[identity] = current
    return digest(identities)


class _WithoutDocstrings(ast.NodeTransformer):
    def _body(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    visit_Module = _body
    visit_ClassDef = _body
    visit_FunctionDef = _body
    visit_AsyncFunctionDef = _body


def _python_tree(source):
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Reviewed Python source is invalid") from exc


def _code_regions(source, tree):
    """Locate comment tokens and actual docstring interiors, excluding quotes."""
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def token_offset(position):
        return starts[position[0] - 1] + position[1]

    doc_positions = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
        ):
            expr = node.body[0].value
            if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                # AST columns count UTF-8 bytes; tokenize columns count characters.
                column = len(lines[expr.lineno - 1].encode()[: expr.col_offset].decode())
                doc_positions.add((expr.lineno, column))
    comments, docstrings = [], []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        start, end = token_offset(token.start), token_offset(token.end)
        if token.type == tokenize.COMMENT:
            comments.append((start, end))
        elif token.type == tokenize.STRING and token.start in doc_positions:
            match = re.match(r"(?i:[rub]*)('''|\"\"\"|'|\")", token.string)
            if match:
                docstrings.append((start + match.end(), end - len(match.group(1))))
    return comments, docstrings


def _validate_code(source, edited, spans):
    before, after = _python_tree(source), _python_tree(edited)
    comments, docstrings = _code_regions(source, before)
    for span in spans:
        start, end, kind = span["start"], span["end"], span["kind"]
        if kind == "docstring_content":
            valid = any(a <= start < end <= b for a, b in docstrings)
        elif kind == "comment":
            valid = all(
                source[i].isspace() or any(a <= i < b for a, b in comments)
                for i in range(start, end)
            )
        else:
            valid = False
        if not valid or span.get("preserve_executable_ast") is not True:
            raise ValueError("Code deletion must be confined to attested comments/docstrings")
    normalized_before = ast.dump(_WithoutDocstrings().visit(before), include_attributes=False)
    normalized_after = ast.dump(_WithoutDocstrings().visit(after), include_attributes=False)
    if normalized_before != normalized_after:
        raise ValueError("Source edit changes executable Python AST")


def _edit_string(source, review):
    spans = review.get("redactions", [])
    if not isinstance(spans, list):
        raise TypeError("Redactions must be a list")
    status = review["review_status"]
    if status != "redact_nonexecuting_text" and spans:
        raise ValueError("Only a redaction decision may contain spans")
    if status == "redact_nonexecuting_text" and not spans:
        raise ValueError("A redaction decision requires exact spans")
    for span in spans:
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source):
            raise ValueError("Invalid source redaction bounds")
        if text_digest(source[start:end]) != span.get("span_sha256"):
            raise ValueError("Stale source redaction span hash")
        _nonempty(span.get("reason"), "redaction reason")
        _nonempty(span.get("kind"), "redaction kind")
        if "text" in span and span["text"] != source[start:end]:
            raise ValueError("Redaction text differs from exact span")
    spans = sorted(spans, key=lambda s: s["start"])
    if any(a["end"] > b["start"] for a, b in pairwise(spans)):
        raise ValueError("Overlapping source redaction spans")
    edited = source
    positions = list(range(len(source)))
    for span in reversed(spans):
        start, end = span["start"], span["end"]
        edited = edited[:start] + edited[end:]
        del positions[start:end]
    expected = None if status == "remove_field" else edited
    if "retained_value" in review and review["retained_value"] != expected:
        raise ValueError("Retained value differs from exact deletion result")
    if "retained_text" in review and review["retained_text"] != expected:
        raise ValueError("Retained text differs from exact deletion result")
    expected_hash = None if status == "remove_field" else digest(expected)
    if "retained_sha256" in review and review["retained_sha256"] != expected_hash:
        raise ValueError("Retained value hash differs from exact deletion result")
    return edited, positions, spans


def _surviving_reviewed_signal(finding, positions, approved):
    """Do not silently approve new scanner matches introduced by concatenation."""
    spans = finding.get("spans", [])
    if not spans:
        return False
    for span in spans:
        start, end = span["start"], span["end"]
        if not 0 <= start < end <= len(positions):
            return False
        old_start, old_end = positions[start], positions[end - 1] + 1
        if old_end - old_start != end - start:
            return False
        if (finding["kind"], old_start, old_end, span["sha256"]) not in approved:
            return False
    return True


def _supersessions(documents):
    """Only an explicit replacement of an unresolved identical source is allowed."""
    reviews = [r for item in documents for r in item.get("review_document", item)["reviews"]]
    indexed = {}
    for review in reviews:
        if "review_id" in review:
            identity = _nonempty(review["review_id"], "review_id")
            if identity in indexed:
                raise ValueError("Ambiguous duplicate review ID")
            indexed[identity] = review

    def occurrence_keys(review):
        return sorted(
            (o["step_id"], o["path"], tuple(sorted(o["finding_ids"])))
            for o in review["occurrences"]
        )

    superseded = {}
    for review in reviews:
        if "supersedes_review_id" not in review:
            continue
        target = review["supersedes_review_id"]
        original = indexed.get(target)
        if (
            original is None
            or target in superseded
            or original["review_status"] != "needs_review"
            or "supersedes_review_id" in original
            or review.get("review_id") not in indexed
            or review["review_status"] == "needs_review"
            or any(review[k] != original[k] for k in ("source_sha256", "raw_text_sha256"))
            or occurrence_keys(review) != occurrence_keys(original)
        ):
            raise ValueError("Invalid or ambiguous source-review supersession")
        superseded[target] = review["review_id"]
    return superseded


def apply_source_reviews(
    compiled_steps, documents, *, source_inventory_sha256, compiler_sha256=None
):
    """Apply trusted fragment artifacts to compiler drafts, atomically in memory.

    All artifact occurrences must identify supplied steps; passing a subset of
    the reviewed inventory fails rather than ignoring unknown occurrences.
    Unchanged unreviewed code is preserved. Surviving scanner matches are closed
    only when mapped to exact previously reviewed spans; novel matches remain.
    """
    inventory_hash = _hash(source_inventory_sha256, "inventory")
    compiler_hash = hashlib.sha256(
        Path(__file__).with_name("wm_compile.py").read_bytes()
    ).hexdigest()
    if compiler_sha256 is not None and _hash(compiler_sha256, "compiler") != compiler_hash:
        raise ValueError("Expected compiler hash differs from current compiler source")
    results = copy.deepcopy(list(compiled_steps))
    by_step = {r["step_id"]: r for r in results}
    if len(by_step) != len(results):
        raise ValueError("Ambiguous duplicate step IDs")
    for result in results:
        if result.get("agent_payload") is not None:
            raise ValueError("Expected private compiler draft, not an approved payload")
        if result["audit"]["draft_input_sha256"] != digest(result["draft_input"]):
            raise ValueError("Stale compiler draft hash")
    documents = list(documents)
    superseded = _supersessions(documents)
    operations, seen, artifact_provenance = [], {}, []
    for item in documents:
        document = item.get("review_document", item)
        document_hash = digest(document)
        if "review_document" in item and item.get("document_sha256") != document_hash:
            raise ValueError("Review document changed after loading")
        if (
            document.get("schema") not in SCOPES
            or document.get("scope") != SCOPES[document["schema"]]
        ):
            raise ValueError("Unsupported review schema or scope")
        if document.get("source_inventory_sha256") != inventory_hash:
            raise ValueError("Stale review inventory hash")
        if document.get("compiler_sha256") != compiler_hash:
            raise ValueError("Stale review compiler hash")
        reviewer = _nonempty(document.get("reviewer"), "reviewer")
        artifact_hash = item.get("artifact_sha256", document_hash)
        _hash(artifact_hash, "artifact")
        artifact_provenance.append(
            {
                "artifact_path": item.get("artifact_path"),
                "artifact_sha256": artifact_hash,
                "document_sha256": document_hash,
                "reviewer": reviewer,
                "review_ids": [r["review_id"] for r in document["reviews"] if "review_id" in r],
            }
        )
        for review in document["reviews"]:
            status = review.get("review_status")
            if status not in STATUSES:
                raise ValueError("Unsupported source review status")
            _nonempty(review.get("rationale"), "rationale")
            if status != "needs_review" and review.get("full_source_read") is not True:
                raise ValueError("Source review requires explicit full-source reading")
            if not review.get("occurrences"):
                raise ValueError("Source review requires explicit occurrences")
            for occurrence in review["occurrences"]:
                step_id, path = occurrence.get("step_id"), occurrence.get("path")
                if step_id not in by_step:
                    raise ValueError("Unknown source occurrence step")
                previous = seen.get((step_id, path))
                if previous is not None and not (
                    (
                        previous.get("review_id") in superseded
                        and superseded[previous["review_id"]] == review.get("review_id")
                    )
                    or (
                        review.get("review_id") in superseded
                        and superseded[review["review_id"]] == previous.get("review_id")
                    )
                ):
                    raise ValueError("Ambiguous duplicate source occurrence")
                if previous is not None and previous is review:
                    raise ValueError("Ambiguous duplicate source occurrence")
                seen[(step_id, path)] = review
                parts = _parts(path)
                is_code = bool(CODE_PATH.fullmatch(path))
                schema = document["schema"]
                if (schema == "wm-code-fragment-review-v1" and not is_code) or (
                    schema == "wm-plan-fragment-review-v1" and is_code
                ):
                    raise ValueError("Source occurrence is outside artifact scope")
                result = by_step[step_id]
                try:
                    source = _lookup(result["draft_input"], parts)
                except (KeyError, IndexError) as exc:
                    raise ValueError("Unknown source occurrence path") from exc
                if not isinstance(source, str):
                    raise TypeError("Only string source fields may be reviewed")
                if digest(source) != review.get("source_sha256") or text_digest(
                    source
                ) != review.get("raw_text_sha256"):
                    raise ValueError("Stale reviewed source hash")
                existing = {
                    f["finding_id"]: f
                    for f in result["audit"]["unresolved_findings"]
                    if f["path"] == path and f["source_sha256"] == digest(source)
                }
                ids = occurrence.get("finding_ids", [])
                if not ids or len(set(ids)) != len(ids) or set(ids) - existing.keys():
                    raise ValueError("Unknown, stale, or ambiguous occurrence finding IDs")
                if (
                    status in {"redact_nonexecuting_text", "remove_field"}
                    and set(ids) != existing.keys()
                ):
                    raise ValueError("Source edits require all current field finding IDs")
                edited, positions, spans = _edit_string(source, review)
                if status == "remove_field":
                    parent = _lookup(result["draft_input"], parts[:-1])
                    if is_code or not isinstance(parent, dict) or set(ids) != existing.keys():
                        raise ValueError(
                            "Whole-field removal requires a fully reviewed plan dictionary leaf"
                        )
                if is_code and status != "needs_review":
                    script_path = result["draft_input"]["code"][int(parts[1])].get("script_path")
                    if script_path not in review.get("script_paths", []):
                        raise ValueError("Unknown code script-path occurrence")
                    _validate_code(source, edited, spans)
                if review.get("review_id") in superseded:
                    continue
                operations.append(
                    {
                        "result": result,
                        "parts": parts,
                        "path": path,
                        "source": source,
                        "edited": edited,
                        "positions": positions,
                        "spans": spans,
                        "review": review,
                        "ids": ids,
                        "existing": existing,
                        "provenance": {
                            "reviewer": reviewer,
                            "artifact_sha256": artifact_hash,
                            "document_sha256": document_hash,
                            "artifact_path": item.get("artifact_path"),
                            "review_id": review.get("review_id"),
                            "supersedes_review_id": review.get("supersedes_review_id"),
                        },
                    }
                )
    for op in operations:
        _apply_operation(op)
    for result in results:
        audit = result["audit"]
        audit["draft_input_sha256"] = digest(result["draft_input"])
        audit["semantic_certificate"] = False
        audit["requires_whole_step_content_review"] = True
        audit["finding_counts"] = dict(Counter(f["kind"] for f in audit["unresolved_findings"]))
        if result["status"] != "rejected":
            result["status"] = "needs_review" if audit["unresolved_findings"] else "draft_clear"
        result["agent_payload"] = None
    return results, {
        "schema": SCHEMA,
        "steps": len(results),
        "reviewed_occurrences": len(operations),
        "review_status_counts": dict(Counter(op["review"]["review_status"] for op in operations)),
        "counts": dict(Counter(r["status"] for r in results)),
        "residual_findings": sum(len(r["audit"]["unresolved_findings"]) for r in results),
        "semantic_certificate": False,
        "source_inventory_sha256": inventory_hash,
        "compiler_sha256": compiler_hash,
        "superseded_reviews": superseded,
        "review_artifacts": artifact_provenance,
    }


def _apply_operation(op):
    result, review = op["result"], op["review"]
    audit, status = result["audit"], review["review_status"]
    source, edited, path = op["source"], op["edited"], op["path"]
    record = {
        **op["provenance"],
        "path": path,
        "review_status": status,
        "source_sha256": digest(source),
        "raw_text_sha256": text_digest(source),
        "edited_sha256": None if status == "remove_field" else digest(edited),
        "finding_ids": list(op["ids"]),
        "rationale": review["rationale"],
        "redactions": copy.deepcopy(op["spans"]),
        "whole_step_approval": False,
    }
    if op.get("manual_source_review"):
        record["manual_source_review"] = True
        record["evidence_sha256"] = review["evidence_sha256"]
        record["source_drafts_sha256"] = op["source_drafts_sha256"]
    audit.setdefault("source_fragment_reviews", []).append(record)
    audit.setdefault("source_hash_mapping", []).append(
        {
            "path": path,
            "source_sha256": record["source_sha256"],
            "edited_sha256": record["edited_sha256"],
            "review_status": status,
        }
    )
    if status == "needs_review":
        return
    if status in {"approve_findings", "approve_source_field"}:
        audit["unresolved_findings"] = [
            f for f in audit["unresolved_findings"] if f["finding_id"] not in op["ids"]
        ]
        record["residual_finding_ids"] = [
            f["finding_id"] for f in audit["unresolved_findings"] if f["path"] == path
        ]
        return
    parent = _lookup(result["draft_input"], op["parts"][:-1])
    key = int(op["parts"][-1]) if isinstance(parent, list) else op["parts"][-1]
    if status == "remove_field":
        del parent[key]
    else:
        parent[key] = edited
    unresolved = [f for f in audit["unresolved_findings"] if f["path"] != path]
    if status != "remove_field":
        if CODE_PATH.fullmatch(path):
            code = result["draft_input"]["code"][int(op["parts"][1])]
            rescanned = list(
                _code_findings(result["step_id"], path, edited, code.get("script_path"))
            )
        else:
            rescanned = list(_scan_operations(result["step_id"], edited, op["parts"]))
        approved = {
            (f["kind"], s["start"], s["end"], s["sha256"])
            for fid in op["ids"]
            for f in [op["existing"][fid]]
            for s in f.get("spans", [])
        }
        approved.update(op.get("previously_reviewed_signals", set()))
        residual = [
            f for f in rescanned if not _surviving_reviewed_signal(f, op["positions"], approved)
        ]
        unresolved.extend(residual)
        record["rescanned_finding_ids"] = [f["finding_id"] for f in rescanned]
        record["residual_finding_ids"] = [f["finding_id"] for f in residual]
    audit["unresolved_findings"] = unresolved
    if status in {"redact_nonexecuting_text", "remove_field"}:
        audit["removed"].append(
            {
                "path": path,
                "kind": "reviewed_source_fragment_deletion",
                "source_sha256": digest(source),
                "content": source,
                "spans": copy.deepcopy(op["spans"]),
                **op["provenance"],
            }
        )


def _previously_reviewed_signals(result, path, source):
    """Carry only provably closed scanner spans through a second-stage edit."""
    source_hash = digest(source)
    known_ids = set()
    for record in result["audit"].get("source_fragment_reviews", []):
        if record["path"] != path:
            continue
        if (
            record.get("source_sha256") == source_hash
            and record["review_status"] == "approve_findings"
        ):
            known_ids.update(record.get("finding_ids", []))
        if record.get("edited_sha256") == source_hash:
            known_ids.update(
                set(record.get("rescanned_finding_ids", []))
                - set(record.get("residual_finding_ids", []))
            )
    for decision in result["audit"].get("applied_decisions", []):
        if decision.get("source_sha256") == source_hash and decision.get("action") == "approve":
            known_ids.add(decision["finding_id"])
    findings = [
        f
        for f in result["audit"].get("findings", [])
        if f["path"] == path and f["source_sha256"] == source_hash
    ]
    if CODE_PATH.fullmatch(path):
        code = result["draft_input"]["code"][int(_parts(path)[1])]
        findings.extend(_code_findings(result["step_id"], path, source, code.get("script_path")))
    else:
        findings.extend(_scan_operations(result["step_id"], source, _parts(path)))
    return {
        (f["kind"], s["start"], s["end"], s["sha256"])
        for f in findings
        if f["finding_id"] in known_ids
        for s in f.get("spans", [])
    }


def apply_manual_source_reviews(
    reviewed_steps, documents, *, source_inventory_sha256, compiler_sha256=None
):
    """Explicit second-stage complete-source review, separate from scanner IDs.

    Artifacts bind the entire current draft map and each exact string field.
    Occurrences must OMIT finding_ids. Fields with CURRENT unresolved scanner
    findings are rejected: resolve those through apply_source_reviews first.
    Previously closed scanner signals remain closed only with recorded evidence;
    newly introduced matches require review. No whole-step approval is created.
    """
    inventory_hash = _hash(source_inventory_sha256, "inventory")
    compiler_hash = hashlib.sha256(
        Path(__file__).with_name("wm_compile.py").read_bytes()
    ).hexdigest()
    if compiler_sha256 is not None and _hash(compiler_sha256, "compiler") != compiler_hash:
        raise ValueError("Expected compiler hash differs from current compiler source")
    results = copy.deepcopy(list(reviewed_steps))
    stage_hash = review_drafts_digest(results)
    by_step = {step["step_id"]: step for step in results}
    if any(step.get("agent_payload") is not None for step in results):
        raise ValueError("Expected private reviewed drafts, not approved agent payloads")
    operations, seen, review_ids, artifacts = [], set(), set(), []
    for item in documents:
        document = item.get("review_document", item)
        document_hash = digest(document)
        if "review_document" in item and document_hash != item.get("document_sha256"):
            raise ValueError("Review document changed after loading")
        if document.get("schema") != MANUAL_SCHEMA or document.get("scope") != MANUAL_SCOPE:
            raise ValueError("Explicit manual-source schema and scope required")
        if document.get("source_inventory_sha256") != inventory_hash:
            raise ValueError("Stale manual review inventory hash")
        if document.get("compiler_sha256") != compiler_hash:
            raise ValueError("Stale manual review compiler hash")
        if document.get("source_drafts_sha256") != stage_hash:
            raise ValueError("Stale manual review current draft-stage hash")
        reviewer = _nonempty(document.get("reviewer"), "reviewer")
        artifact_hash = _hash(item.get("artifact_sha256", document_hash), "artifact")
        artifact = {
            "artifact_path": item.get("artifact_path"),
            "artifact_sha256": artifact_hash,
            "document_sha256": document_hash,
            "reviewer": reviewer,
        }
        artifacts.append(artifact)
        for review in document["reviews"]:
            identity = _nonempty(review.get("review_id"), "review_id")
            if identity in review_ids or "supersedes_review_id" in review:
                raise ValueError("Ambiguous manual review ID or unsupported supersession")
            review_ids.add(identity)
            status = review.get("review_status")
            if status not in {
                "approve_source_field",
                "redact_nonexecuting_text",
                "remove_field",
                "needs_review",
            }:
                raise ValueError("Unsupported manual source-field review status")
            _nonempty(review.get("rationale"), "rationale")
            _hash(review.get("evidence_sha256"), "manual evidence")
            if status != "needs_review" and not all(
                review.get(key) is True
                for key in ("full_source_read", "outcome_free", "executable_recipe_preserved")
            ):
                raise ValueError(
                    "Manual source action requires full-source read and both attestations"
                )
            if not review.get("occurrences"):
                raise ValueError("Manual source review requires explicit occurrences")
            for occurrence in review["occurrences"]:
                if "finding_ids" in occurrence or "finding_ids" in review:
                    raise ValueError(
                        "Manual source reviews must not pretend scanner finding IDs exist"
                    )
                step_id, path = occurrence.get("step_id"), occurrence.get("path")
                if step_id not in by_step:
                    raise ValueError("Unknown manual source occurrence step")
                if (step_id, path) in seen:
                    raise ValueError("Ambiguous duplicate manual source occurrence")
                seen.add((step_id, path))
                result, parts = by_step[step_id], _parts(path)
                try:
                    source = _lookup(result["draft_input"], parts)
                except (KeyError, IndexError) as exc:
                    raise ValueError("Unknown manual source occurrence path") from exc
                if not isinstance(source, str):
                    raise TypeError("Manual reviews require complete string source fields")
                if digest(source) != review.get("source_sha256") or text_digest(
                    source
                ) != review.get("raw_text_sha256"):
                    raise ValueError("Stale manual reviewed source hash")
                if any(f["path"] == path for f in result["audit"]["unresolved_findings"]):
                    raise ValueError("Manual source field has CURRENT unresolved scanner findings")
                edited, positions, spans = _edit_string(source, review)
                if status == "remove_field" and (
                    CODE_PATH.fullmatch(path)
                    or not isinstance(_lookup(result["draft_input"], parts[:-1]), dict)
                ):
                    raise ValueError("Manual field removal requires a plan dictionary string leaf")
                if CODE_PATH.fullmatch(path) and status != "needs_review":
                    script_path = result["draft_input"]["code"][int(parts[1])].get("script_path")
                    if script_path not in review.get("script_paths", []):
                        raise ValueError("Unknown manual code script-path occurrence")
                    _validate_code(source, edited, spans)
                operations.append(
                    {
                        "result": result,
                        "parts": parts,
                        "path": path,
                        "source": source,
                        "edited": edited,
                        "positions": positions,
                        "spans": spans,
                        "review": review,
                        "ids": [],
                        "existing": {},
                        "manual_source_review": True,
                        "source_drafts_sha256": stage_hash,
                        "previously_reviewed_signals": _previously_reviewed_signals(
                            result, path, source
                        ),
                        "provenance": {**artifact, "review_id": identity},
                    }
                )
    for op in operations:
        _apply_operation(op)
        audit = op["result"]["audit"]
        holds = [
            hold for hold in audit.get("manual_source_holds", []) if hold["path"] != op["path"]
        ]
        if op["review"]["review_status"] == "needs_review":
            holds.append(
                {
                    "path": op["path"],
                    "review_id": op["review"]["review_id"],
                    "source_sha256": digest(op["source"]),
                    "evidence_sha256": op["review"]["evidence_sha256"],
                }
            )
        audit["manual_source_holds"] = holds
    for result in results:
        audit = result["audit"]
        audit["draft_input_sha256"] = digest(result["draft_input"])
        audit["semantic_certificate"] = False
        audit["requires_whole_step_content_review"] = True
        audit["finding_counts"] = dict(Counter(f["kind"] for f in audit["unresolved_findings"]))
        if result["status"] != "rejected":
            result["status"] = (
                "needs_review"
                if audit["unresolved_findings"] or audit.get("manual_source_holds")
                else "draft_clear"
            )
        result["agent_payload"] = None
    return results, {
        "schema": MANUAL_SCHEMA,
        "scope": MANUAL_SCOPE,
        "steps": len(results),
        "source_drafts_sha256": stage_hash,
        "edited_drafts_sha256": review_drafts_digest(results),
        "source_inventory_sha256": inventory_hash,
        "compiler_sha256": compiler_hash,
        "reviewed_occurrences": len(operations),
        "review_artifacts": artifacts,
        "review_status_counts": dict(Counter(op["review"]["review_status"] for op in operations)),
        "counts": dict(Counter(step["status"] for step in results)),
        "residual_findings": sum(len(step["audit"]["unresolved_findings"]) for step in results),
        "manual_holds": sum(len(step["audit"].get("manual_source_holds", [])) for step in results),
        "semantic_certificate": False,
    }


def observed_numeric_source_digest(value):
    """Hash an exact JSON numeric type, without treating bool as int or zero as absence.

    Integers are mathematically finite; do not convert them to float to check
    finiteness (large integers can overflow). JSON encoding limits still apply.
    """
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("Observed metadata must be a finite int or float, never bool")
    return digest({"type": type(value).__name__, "value": value})


def _numeric_exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f"Unexpected or missing {name} fields")


def _numeric_leaf(draft, path):
    if not isinstance(path, str) or not NUMERIC_PATH.fullmatch(path):
        raise ValueError("Only observed data n_examples or progress total may be deleted")
    parts = path[1:].split("/")
    try:
        setup = draft["plan"]["setup"]
        if parts[2] == "data" and not isinstance(setup["data"], list):
            raise ValueError("Observed data index requires a list, not a numeric dictionary key")
        parent = _lookup(draft, parts[:-1])
        if not isinstance(parent, dict):
            raise TypeError("Observed metadata must be a dictionary numeric leaf")
        value = parent[parts[-1]]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unknown observed numeric metadata path") from exc
    observed_numeric_source_digest(value)
    return parent, parts[-1], value


def _numeric_supporting_source(draft, source):
    if not isinstance(source, dict):
        raise TypeError("Numeric evidence source must be an explicit record")
    if source.get("kind") == "draft_text":
        _numeric_exact_keys(
            source,
            "kind path source_sha256 raw_text_sha256 start end text span_sha256",
            "draft evidence source",
        )
        parts = _parts(source["path"])
        try:
            content = _lookup(draft, parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Unknown numeric evidence source path") from exc
        if not isinstance(content, str):
            raise ValueError("Numeric evidence must bind a complete plan or code string")
        if (
            digest(content) != source["source_sha256"]
            or text_digest(content) != source["raw_text_sha256"]
        ):
            raise ValueError("Stale numeric evidence source text hash")
        start, end = source["start"], source["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(content):
            raise ValueError("Invalid numeric evidence source span")
        span = content[start:end]
        if span != source["text"] or text_digest(span) != source["span_sha256"]:
            raise ValueError("Altered numeric evidence source span")
    elif source.get("kind") == "private_text":
        _numeric_exact_keys(
            source,
            "kind source_locator source_artifact_sha256 content content_sha256",
            "private evidence source",
        )
        _nonempty(source["source_locator"], "private evidence locator")
        _hash(source["source_artifact_sha256"], "private evidence artifact")
        content = _nonempty(source["content"], "private evidence content")
        if text_digest(content) != source["content_sha256"]:
            raise ValueError("Altered private numeric evidence content")
    else:
        raise ValueError("Unsupported numeric evidence source kind")


def _validate_numeric_evidence(result, review):
    evidence = review["evidence"]
    _numeric_exact_keys(
        evidence,
        "step_id source_step_sha256 source_projection_sha256 classification "
        "observed_not_configured_budget supporting_sources",
        "numeric evidence",
    )
    if digest(evidence) != _hash(review["evidence_sha256"], "numeric evidence"):
        raise ValueError("Altered numeric evidence hash")
    if (
        evidence["step_id"] != result["step_id"]
        or evidence["source_step_sha256"] != review["source_step_sha256"]
    ):
        raise ValueError("Numeric evidence must bind the exact whole source step")
    projection_hash = result["audit"].get("source_recipe_projection_sha256") or result["audit"].get(
        "source_input_sha256"
    )
    if evidence["source_projection_sha256"] != _hash(projection_hash, "source projection"):
        raise ValueError("Stale numeric evidence source provenance")
    allowed = (
        {"observed_data_yield"}
        if review["path"].endswith("/n_examples")
        else {"observed_dependent_schedule", "observed_achieved_progress"}
    )
    if (
        not isinstance(evidence["classification"], str)
        or evidence["classification"] not in allowed
        or evidence["observed_not_configured_budget"] is not True
    ):
        raise ValueError("Explicit observed, not configured-budget, evidence is required")
    sources = evidence["supporting_sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("Numeric review needs bound supporting source evidence")
    for source in sources:
        _numeric_supporting_source(result["draft_input"], source)


def _numeric_reject_mutable_aliases(value):
    """Require a JSON tree: deepcopy would preserve aliases and broaden deletions."""
    seen = set()

    def visit(item):
        if isinstance(item, (dict, list)):
            if id(item) in seen:
                raise ValueError("Aliased or cyclic mutable numeric-review inputs are ambiguous")
            seen.add(id(item))
            children = item.values() if isinstance(item, dict) else item
            for child in children:
                visit(child)
        elif isinstance(item, tuple):
            for child in item:
                visit(child)

    visit(value)


def apply_observed_numeric_reviews(
    reviewed_steps, documents, *, source_inventory_sha256, compiler_sha256=None
):
    """Delete only explicitly reviewed observed numeric metadata, never replace it.

    This is separate from both string review APIs. Every decision binds the
    entire current draft map and its complete source step, exact typed value,
    reviewer attestations, and private evidence. Classification is a trusted
    human decision, not an inference made here. Private text locators are
    reviewer-supplied provenance: they are never opened, and membership in an
    external artifact is not independently certified by this integrator.

    All existing holds and scanner findings survive, including holds on a
    deleted leaf. Deletion grants neither whole-step nor execution approval.
    Evidence and removed values live only in private audit, not draft_input.
    """
    inventory_hash = _hash(source_inventory_sha256, "inventory")
    compiler_hash = hashlib.sha256(
        Path(__file__).with_name("wm_compile.py").read_bytes()
    ).hexdigest()
    if compiler_sha256 is not None and _hash(compiler_sha256, "compiler") != compiler_hash:
        raise ValueError("Expected compiler hash differs from current compiler source")
    input_steps = list(reviewed_steps)
    _numeric_reject_mutable_aliases(input_steps)
    results = copy.deepcopy(input_steps)
    stage_hash = review_drafts_digest(results)
    by_step = {step["step_id"]: step for step in results}
    if any(step.get("agent_payload") is not None for step in results):
        raise ValueError("Expected private reviewed drafts, not approved agent payloads")
    documents = list(documents)
    if not documents:
        raise ValueError("Explicit observed numeric review artifacts are required")
    operations, seen, review_ids, artifacts = [], set(), set(), []
    for item in documents:
        if not isinstance(item, dict):
            raise TypeError("Numeric review document must be an object")
        document = item.get("review_document", item)
        if not isinstance(document, dict):
            raise TypeError("Numeric review document must be an object")
        document_hash = digest(document)
        if "review_document" in item and document_hash != item.get("document_sha256"):
            raise ValueError("Review document changed after loading")
        if document.get("schema") != NUMERIC_SCHEMA or document.get("scope") != NUMERIC_SCOPE:
            raise ValueError("Explicit observed-numeric schema and scope required")
        for key, expected in (
            ("source_inventory_sha256", inventory_hash),
            ("compiler_sha256", compiler_hash),
            ("source_drafts_sha256", stage_hash),
        ):
            if document.get(key) != expected:
                raise ValueError(f"Stale numeric review {key}")
        artifact = {
            "artifact_path": item.get("artifact_path"),
            "artifact_sha256": _hash(item.get("artifact_sha256", document_hash), "artifact"),
            "document_sha256": document_hash,
            "reviewer": _nonempty(document.get("reviewer"), "reviewer"),
        }
        artifacts.append(artifact)
        reviews = document.get("reviews")
        if not isinstance(reviews, list) or not reviews:
            raise ValueError("Numeric review document requires explicit decisions")
        for review in reviews:
            _numeric_exact_keys(
                review,
                "review_id step_id path review_status source_step_sha256 source_type "
                "source_sha256 full_source_read outcome_free executable_recipe_preserved "
                "rationale evidence_sha256 evidence",
                "numeric review",
            )
            identity = _nonempty(review["review_id"], "review_id")
            if identity in review_ids:
                raise ValueError("Ambiguous duplicate numeric review ID")
            review_ids.add(identity)
            if review["review_status"] != "delete_observed_numeric_metadata":
                raise ValueError("Numeric review permits deletion only")
            if not all(
                review[key] is True
                for key in ("full_source_read", "outcome_free", "executable_recipe_preserved")
            ):
                raise ValueError("Numeric deletion requires full-source read and both attestations")
            _nonempty(review["rationale"], "numeric review rationale")
            step_id = _nonempty(review["step_id"], "numeric step_id")
            if step_id not in by_step:
                raise ValueError("Unknown numeric review source step")
            result = by_step[step_id]
            parent, key, value = _numeric_leaf(result["draft_input"], review["path"])
            if (step_id, review["path"]) in seen:
                raise ValueError("Ambiguous duplicate numeric source path")
            seen.add((step_id, review["path"]))
            if review["source_step_sha256"] != digest(result["draft_input"]):
                raise ValueError("Stale numeric review whole source step hash")
            if review["source_type"] != type(value).__name__ or review[
                "source_sha256"
            ] != observed_numeric_source_digest(value):
                raise ValueError("Stale typed numeric source hash")
            _validate_numeric_evidence(result, review)
            operations.append((result, parent, key, value, review, artifact))

    # Validate every operation before applying any. Caller-owned objects are never mutated.
    for result, parent, key, value, review, artifact in operations:
        del parent[key]
        result["audit"].setdefault("observed_numeric_metadata_reviews", []).append(
            {
                **copy.deepcopy(review),
                "removed_value": value,
                "source_drafts_sha256": stage_hash,
                "provenance": copy.deepcopy(artifact),
            }
        )
    for result in results:
        audit = result["audit"]
        audit["draft_input_sha256"] = digest(result["draft_input"])
        audit["semantic_certificate"] = False
        audit["requires_whole_step_content_review"] = True
        if result["status"] != "rejected" and (
            audit.get("unresolved_findings") or audit.get("manual_source_holds")
        ):
            result["status"] = "needs_review"
        result["agent_payload"] = None
    return results, {
        "schema": NUMERIC_SCHEMA,
        "scope": NUMERIC_SCOPE,
        "steps": len(results),
        "source_drafts_sha256": stage_hash,
        "edited_drafts_sha256": review_drafts_digest(results),
        "source_inventory_sha256": inventory_hash,
        "compiler_sha256": compiler_hash,
        "deleted_numeric_leaves": len(operations),
        "review_artifacts": artifacts,
        "counts": dict(Counter(step["status"] for step in results)),
        "manual_holds": sum(len(step["audit"].get("manual_source_holds", [])) for step in results),
        "semantic_certificate": False,
        "approved_recipe_count": 0,
        "automatic_numeric_classification": False,
    }


def compile_reviewed_steps(rows, documents, *, source_inventory_sha256, compiler_sha256=None):
    """Project task/plan/code only, compile, and apply explicit review artifacts."""
    projected = []
    for row in rows:
        source = row["model_input"]
        projected.append(
            {
                "example_id": row["example_id"],
                "model_input": {
                    key: copy.deepcopy(source.get(key, {} if key != "code" else []))
                    for key in ("task", "plan", "code")
                },
            }
        )
    compiled, _ = compile_steps(projected)
    for result, row in zip(compiled, projected, strict=True):
        result["audit"]["source_recipe_projection_sha256"] = digest(row["model_input"])
    return apply_source_reviews(
        compiled,
        documents,
        source_inventory_sha256=source_inventory_sha256,
        compiler_sha256=compiler_sha256,
    )


def build_artifact(inventory_path, review_paths, output_dir):
    """Persist a new PRIVATE draft/audit bundle, never overwrite existing work.

    Only the explicitly supplied review artifacts are active. This function
    does not discover supplemental reviews or resolve their target-scope holds.
    Removed outcome text remains in the private audit, never an agent payload.
    """
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing review bundle")
    inventory = Path(inventory_path)
    raw = inventory.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not review_paths:
        raise ValueError("Explicit active review artifacts are required")
    documents = load_review_documents(review_paths)
    steps, summary = compile_reviewed_steps(
        rows, documents, source_inventory_sha256=hashlib.sha256(raw).hexdigest()
    )
    files = {
        "steps.json": steps,
        "audit.json": summary,
    }
    encoded = {
        name: (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
        for name, value in files.items()
    }
    provenance = {
        "schema": SCHEMA,
        "visibility": "private_not_agent_evidence",
        "inventory_path": str(inventory),
        "inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "integration_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler_sha256": summary["compiler_sha256"],
        "active_review_artifacts": summary["review_artifacts"],
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in encoded.items()},
        "semantic_certificate": False,
        "approved_recipe_count": 0,
        "limits": [
            "Only explicit active reviews were applied; no automatic supplemental review discovery.",
            "Audit data retains outcome text and must never be mounted as agent evidence.",
            "Fragment/scanner clearance is not full-step approval, target binding, or complete lineage.",
            "No labels were used for compilation and no model was fitted or evaluated.",
        ],
    }
    encoded["provenance.json"] = (
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    output.mkdir(parents=True, mode=0o700)
    for name, content in encoded.items():
        path = output / name
        with path.open("xb") as handle:
            handle.write(content)
        path.chmod(0o600)
    return summary


def build_manual_artifact(source_bundle, review_paths, output_dir):
    """Persist an explicit second-stage PRIVATE bundle without re-reading labels."""
    source, output = Path(source_bundle), Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing review bundle")
    raw = (source / "steps.json").read_bytes()
    source_provenance_raw = (source / "provenance.json").read_bytes()
    source_provenance = json.loads(source_provenance_raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    if source_provenance.get("files", {}).get("steps.json") != source_hash:
        raise ValueError("Stale source bundle steps hash")
    if not review_paths:
        raise ValueError("Explicit manual review artifacts are required")
    documents = load_review_documents(review_paths)
    for item in documents:
        declared = item["review_document"].get("source_drafts_file_sha256")
        if declared is not None and declared != source_hash:
            raise ValueError("Manual artifact source bundle file hash differs")
    inventory_hash = source_provenance.get("inventory_sha256")
    steps, summary = apply_manual_source_reviews(
        json.loads(raw), documents, source_inventory_sha256=inventory_hash
    )
    files = {"steps.json": steps, "audit.json": summary}
    encoded = {
        name: (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
        for name, value in files.items()
    }
    provenance = {
        "schema": MANUAL_SCHEMA,
        "visibility": "private_not_agent_evidence",
        "source_bundle": str(source),
        "source_steps_sha256": source_hash,
        "source_bundle_provenance_sha256": hashlib.sha256(source_provenance_raw).hexdigest(),
        "source_drafts_sha256": summary["source_drafts_sha256"],
        "inventory_path": source_provenance.get("inventory_path"),
        "inventory_sha256": inventory_hash,
        "compiler_sha256": summary["compiler_sha256"],
        "integration_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "active_review_artifacts": summary["review_artifacts"],
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in encoded.items()},
        "semantic_certificate": False,
        "approved_recipe_count": 0,
        "limits": [
            "Explicit current-source field review only; no whole-step or execution approval.",
            "Source private audit and its approvals remain preserved through the pinned preceding bundle.",
            "No inventory labels were loaded and no model was fitted or evaluated.",
        ],
    }
    encoded["provenance.json"] = (
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    output.mkdir(parents=True, mode=0o700)
    for name, content in encoded.items():
        path = output / name
        with path.open("xb") as handle:
            handle.write(content)
        path.chmod(0o600)
    return summary


def _numeric_json_loads(raw):
    """Strict JSON for the NEW numeric interface only; old loaders are unchanged."""

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key in numeric review input")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("Nonfinite JSON number in numeric review input")

    result = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    # Also rejects finite-looking JSON exponents that overflow float, e.g. 1e999.
    json.dumps(result, allow_nan=False)
    return result


def _numeric_no_symlinks(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlinks are not accepted for numeric review bundle paths")
    return path


def build_observed_numeric_artifact(source_bundle, review_paths, output_dir):
    """Write a new private numeric-deletion stage; never load inventory or labels.

    Requires each review to pin the raw current steps and provenance files as
    well as the canonical draft map. Source steps AND audit bytes must match
    the preceding bundle manifest. Existing string builders/CLI are unchanged;
    callers must opt into this separate Python API. No extra files are found
    by globbing, and private evidence locators are never dereferenced.
    """
    output = _numeric_no_symlinks(output_dir)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing review bundle")
    source = _numeric_no_symlinks(source_bundle)
    source_files = {}
    for name in ("steps.json", "audit.json", "provenance.json"):
        path = _numeric_no_symlinks(source / name)
        source_files[name] = path.read_bytes()
    provenance_raw = source_files["provenance.json"]
    source_provenance = _numeric_json_loads(provenance_raw)
    if not isinstance(source_provenance, dict) or set(source_provenance.get("files", {})) != {
        "steps.json",
        "audit.json",
    }:
        raise ValueError("Numeric source bundle requires exact steps and audit file manifest")
    source_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in source_files.items()}
    for name in ("steps.json", "audit.json"):
        if source_provenance["files"][name] != source_hashes[name]:
            raise ValueError(f"Stale numeric source bundle {name} hash")
    steps = _numeric_json_loads(source_files["steps.json"])
    _numeric_json_loads(source_files["audit.json"])
    inventory_hash = _hash(source_provenance.get("inventory_sha256"), "source inventory")
    compiler_hash = _hash(source_provenance.get("compiler_sha256"), "source compiler")
    # Prior integration code can differ: this stage extends, rather than rewrites, that stage.
    _hash(source_provenance.get("integration_source_sha256"), "source integration")
    documents = []
    for path in review_paths:
        path = _numeric_no_symlinks(path)
        raw = path.read_bytes()
        document = _numeric_json_loads(raw)
        if (
            not isinstance(document, dict)
            or document.get("source_drafts_file_sha256") != source_hashes["steps.json"]
        ):
            raise ValueError("Numeric artifact source steps file hash differs")
        if document.get("source_bundle_provenance_sha256") != source_hashes["provenance.json"]:
            raise ValueError("Numeric artifact source provenance file hash differs")
        documents.append(
            {
                "review_document": document,
                "artifact_path": str(path),
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "document_sha256": digest(document),
            }
        )
    edited, summary = apply_observed_numeric_reviews(
        steps, documents, source_inventory_sha256=inventory_hash, compiler_sha256=compiler_hash
    )

    def encode(value):
        return (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode()

    encoded = {"steps.json": encode(edited), "audit.json": encode(summary)}
    provenance = {
        "schema": NUMERIC_SCHEMA,
        "scope": NUMERIC_SCOPE,
        "visibility": "private_not_agent_evidence",
        "source_bundle": str(source),
        "source_steps_sha256": source_hashes["steps.json"],
        "source_audit_sha256": source_hashes["audit.json"],
        "source_bundle_provenance_sha256": source_hashes["provenance.json"],
        "source_drafts_sha256": summary["source_drafts_sha256"],
        "inventory_path": source_provenance.get("inventory_path"),
        "inventory_sha256": inventory_hash,
        "compiler_sha256": summary["compiler_sha256"],
        "integration_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "active_review_artifacts": summary["review_artifacts"],
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in encoded.items()},
        "semantic_certificate": False,
        "approved_recipe_count": 0,
        "limits": [
            "Deletion-only explicit observed metadata review, never automatic number removal.",
            "Private reviewer evidence and removed values are not positive model or agent input.",
            "Private evidence artifact membership is reviewer-attested, not independently certified.",
            "All existing source holds survive; this grants no whole-step or execution approval.",
            "No inventory labels were loaded and no model was fitted or evaluated.",
        ],
    }
    encoded["provenance.json"] = encode(provenance)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    for name, raw in encoded.items():
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inventory", type=Path)
    inputs.add_argument("--source-bundle", type=Path)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = (
        build_manual_artifact(args.source_bundle, args.review, args.output_dir)
        if args.source_bundle
        else build_artifact(args.inventory, args.review, args.output_dir)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
