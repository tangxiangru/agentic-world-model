"""Draft outcome-free recipe-step compiler; never a semantic approval authority.

Only ``example_id`` and ``model_input`` are read from an inventory row. Removed
content and findings are PRIVATE audit material, not an agent evidence corpus.
Operational setup/code is never automatically rewritten. Fragment decisions
must be supplied by a trusted reviewer, are bound to the exact source content,
and do not replace the separate whole-step/full-recipe content review.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Literal

SCHEMA = "wm-step-compiler-draft-v1"
PLAN_FIELDS = {"problem", "hypothesis", "setup", "evaluation"}
CODE_FIELDS = {"role", "script_path", "status", "content"}
OUTCOME_FIELDS = {
    "result",
    "results",
    "conclusion",
    "measurements",
    "prior_observations",
    "known_previous_checkpoints",
    "official_accuracy",
    "observed_accuracy",
    "observed_score",
    "official_metric",
    "official_correct_of_30",
    "stderr",
    "delta_vs_comparator",
    "label",
    "labels",
    "y",
}
OBSERVATION_FIELDS = {
    "evidence",
    "failure_examples",
    "observations",
    "diagnostic_result",
    "diagnostic_results",
    "watch_set_result",
    "affected_share",
    "watch_set",
}
METRIC = r"(?:acc(?:uracy)?|scores?|pass[@_-]?1|success[_ ]rate|termination[_ ]rate|loss)"
METRIC_NUMBER = re.compile(
    rf"\b{METRIC}(?=\b|[0-9])\s*(?:(?:is|was|of|at|to|from|equals?)\s*|[:=]\s*)?"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*%?",
    re.IGNORECASE,
)
PAST_RESULT = re.compile(
    r"\b(?:scored|achieved|outperformed|underperformed|regressed|collapsed|crashed|"
    r"plateaued|improved|deteriorated|fell|rose|reached|was adopted|were adopted)\b",
    re.IGNORECASE,
)
IMPLICIT_RESULT = re.compile(
    r"\b(?:incumbent|shipped model|best model|better parent|worse|harmful|neutral|"
    r"failed|failure|did not fix|does not fix|fixed|regression|collapse|truncation|"
    r"remaining errors|still failing|ceiling|same curve)\b",
    re.IGNORECASE,
)
EXPERIMENT_REF = re.compile(
    r"\bexp[-_ ]?\d+(?:/\d+)*\b|\b(?:parent|checkpoint|model)\b", re.IGNORECASE
)
NUMERIC_COMPARISON = re.compile(
    r"\b(?:baseline|incumbent|parent|exp[-_ ]?\d+)\b[^\n.!?]{0,70}"
    r"(?:\d+(?:\.\d+)?\s*%|\d+\s*/\s*\d+)",
    re.IGNORECASE,
)
STATIC_METRIC_NAME = re.compile(rf"(?:^|_){METRIC}(?:$|_)", re.IGNORECASE)
HASH = re.compile(r"[a-f0-9]{64}")


def digest(value):
    """Same canonical JSON digest as wm_recipe_input.digest, without importing it."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class ContentDecision:
    """Trusted, content-bound fragment decision, not a whole-step certificate.

    ``redact`` deletes an exact string span, without inserting replacement text.
    ``remove_field`` explicitly removes a whole flagged operational metadata
    field; list entries must instead be redacted/reviewed so paths stay stable.
    A reviewer must attest both flags for approve/redact/remove_field actions.
    """

    finding_id: str
    source_sha256: str
    action: Literal["approve", "redact", "remove_field", "reject"]
    reviewer: str
    evidence_sha256: str
    rationale: str
    outcome_free: bool = False
    executable_recipe_preserved: bool = False
    start: int | None = None
    end: int | None = None
    span_sha256: str | None = None


def _pointer(parts):
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def _parts(pointer):
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _lookup(value, pointer):
    current = value
    for part in _parts(pointer):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _replace(value, pointer, replacement=None, *, remove=False):
    parts = _parts(pointer)
    current = value
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    last = int(parts[-1]) if isinstance(current, list) else parts[-1]
    if remove:
        if isinstance(current, list):
            raise ValueError("Manual removal of list entries would destabilize audit paths")
        del current[last]
    else:
        current[last] = replacement


def _span(value, start, end):
    return {"start": start, "end": end, "sha256": text_digest(value[start:end])}


def _finding(step_id, path, kind, value, spans=()):
    source_hash = digest(value)
    spans = list(spans)
    identity = {
        "step_id": step_id,
        "path": path,
        "kind": kind,
        "source_sha256": source_hash,
        "spans": spans,
    }
    if isinstance(value, str) and spans:
        start = max(0, spans[0]["start"] - 60)
        snippet = value[start : min(len(value), spans[0]["end"] + 100)]
    else:
        snippet = (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))[:240]
    return {
        **identity,
        "finding_id": digest(identity),
        "snippet": snippet,
        "requested_action": "Trusted reviewer must confirm outcome-free semantics and executable preservation, redact exact bound content, or reject; do not infer approval from this scanner.",
    }


def _text_signals(value, *, code=False):
    matches = [("explicit_metric_literal", m) for m in METRIC_NUMBER.finditer(value)]
    matches += [("numeric_ancestor_comparison", m) for m in NUMERIC_COMPARISON.finditer(value)]
    if (
        not code
        or EXPERIMENT_REF.search(value)
        or re.search(rf"\b{METRIC}\b", value, re.IGNORECASE)
    ):
        matches += [("past_outcome_language", m) for m in PAST_RESULT.finditer(value)]
        matches += [("implicit_outcome_reference", m) for m in IMPLICIT_RESULT.finditer(value)]
    return [(kind, _span(value, m.start(), m.end())) for kind, m in matches]


def _ast_span(source, node):
    # AST columns count UTF-8 bytes; output spans count Python string characters.
    lines = source.splitlines(keepends=True)

    def offset(line, column):
        return sum(map(len, lines[: line - 1])) + len(lines[line - 1].encode()[:column].decode())

    return _span(
        source, offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)
    )


def _static_number(node):
    return any(
        isinstance(n, ast.Constant) and type(n.value) in (int, float) for n in ast.walk(node)
    ) and not any(
        isinstance(n, (ast.Name, ast.Call, ast.Attribute, ast.Subscript, ast.Lambda))
        for n in ast.walk(node)
    )


def _code_findings(step_id, path, value, script_path):
    signals = _text_signals(value, code=True)
    # Strings/comments are scanned conservatively, but a dynamically computed
    # accuracy = correct / total does not establish an observed result literal.
    try:
        tree = ast.parse(value)
    except SyntaxError:
        if isinstance(script_path, str) and script_path.endswith(".py"):
            yield _finding(step_id, path, "python_code_unparseable", value)
    else:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if _static_number(node.value) and any(
                    STATIC_METRIC_NAME.search(ast.unparse(t)) for t in targets
                ):
                    signals.append(("static_metric_assignment", _ast_span(value, node)))
            if isinstance(node, ast.Dict):
                for key, item in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and STATIC_METRIC_NAME.search(key.value)
                        and _static_number(item)
                    ):
                        signals.append(("static_metric_mapping", _ast_span(value, item)))
    grouped = defaultdict(list)
    for kind, span in signals:
        if span not in grouped[kind]:
            grouped[kind].append(span)
    for kind, spans in grouped.items():
        yield _finding(step_id, path, kind, value, spans)


def _scan_operations(step_id, value, parts):
    """Operational fields stay intact, including suspect result-like content."""
    path = _pointer(parts)
    if isinstance(value, dict):
        for key, item in value.items():
            if key in OUTCOME_FIELDS or key in OBSERVATION_FIELDS:
                yield _finding(
                    step_id, _pointer([*parts, key]), "outcome_field_in_operational_setup", item
                )
            yield from _scan_operations(step_id, item, [*parts, key])
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _scan_operations(step_id, item, [*parts, index])
    elif isinstance(value, str):
        grouped = defaultdict(list)
        for kind, span in _text_signals(value):
            grouped[kind].append(span)
        for kind, spans in grouped.items():
            yield _finding(step_id, path, kind, value, spans)


def _clean_plan(step_id, plan, removed, findings):
    def remove(value, parts, reason):
        removed.append(
            {
                "path": _pointer(parts),
                "kind": reason,
                "source_sha256": digest(value),
                "content": copy.deepcopy(value),
                "spans": [_span(value, 0, len(value))] if isinstance(value, str) else [],
            }
        )

    def walk(value, parts):
        if parts[:2] == ["plan", "setup"] or parts[:3] == ["plan", "evaluation", "protocol"]:
            findings.extend(_scan_operations(step_id, value, parts))
            return copy.deepcopy(value)
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                child = [*parts, key]
                forbidden = key in OUTCOME_FIELDS or key in OBSERVATION_FIELDS
                forbidden |= parts == ["plan", "evaluation"] and key in {
                    "diagnostic",
                    "diagnostics",
                }
                forbidden |= parts and parts[-1] == "comparator" and key != "ref"
                if forbidden:
                    remove(item, child, "structured_observation_removed")
                elif isinstance(item, str):
                    signals = _text_signals(item)
                    explicit = any(kind != "implicit_outcome_reference" for kind, _ in signals)
                    if explicit:
                        remove(item, child, "outcome_bearing_narrative_removed")
                    else:
                        result[key] = walk(item, child)
                else:
                    result[key] = walk(item, child)
            return result
        if isinstance(value, list):
            # Keep list positions stable for exact JSON-pointer audit paths.
            return [walk(item, [*parts, index]) for index, item in enumerate(value)]
        if isinstance(value, str):
            for kind, span in _text_signals(value):
                findings.append(_finding(step_id, _pointer(parts), kind, value, [span]))
        return copy.deepcopy(value)

    return walk(plan, ["plan"])


def _validate_decision(decision, finding):
    if decision.source_sha256 != finding["source_sha256"]:
        raise ValueError("Stale manual decision: source content hash differs")
    if not all(
        isinstance(v, str) and v.strip() for v in (decision.reviewer, decision.rationale)
    ) or not HASH.fullmatch(decision.evidence_sha256):
        raise ValueError("Manual decisions require reviewer, rationale, and evidence SHA256")
    if decision.action not in {"approve", "redact", "remove_field", "reject"}:
        raise ValueError("Unsupported manual action")
    if decision.action != "reject" and not (
        decision.outcome_free is True and decision.executable_recipe_preserved is True
    ):
        raise ValueError("Manual change/approval requires both explicit reviewer attestations")
    if decision.action != "redact" and any(
        v is not None for v in (decision.start, decision.end, decision.span_sha256)
    ):
        raise ValueError("Only a redact decision may specify a text span")


def compile_step(row, *, decisions=()):
    """Return a private draft and audit; never return an approved agent payload.

    Only model_input and example_id are accessed. Result/label/history mutations
    outside model_input cannot influence compilation. Manual redactions are
    rescanned, and unresolved new findings require another review pass on that
    draft. All output remains provisional even if no heuristic finding remains.
    """
    step_id = row["example_id"]
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("A nonempty example_id is required")
    source = row["model_input"]
    if not isinstance(source, dict) or not isinstance(source.get("plan"), dict):
        raise TypeError("model_input must contain a plan object")
    # Deliberately do not copy/read the complete inventory row.
    source_hash = digest(source)
    removed, findings = [], []
    plan = {
        key: copy.deepcopy(value) for key, value in source["plan"].items() if key in PLAN_FIELDS
    }
    for key, value in source["plan"].items():
        if key not in PLAN_FIELDS:
            removed.append(
                {
                    "path": _pointer(["plan", key]),
                    "kind": "non_plan_field_removed",
                    "source_sha256": digest(value),
                    "content": copy.deepcopy(value),
                    "spans": [],
                }
            )
    draft = {
        "task": copy.deepcopy(source.get("task", {})),
        "plan": _clean_plan(step_id, plan, removed, findings),
        "code": [],
    }
    findings.extend(_scan_operations(step_id, draft["task"], ["task"]))
    for index, entry in enumerate(source.get("code", [])):
        if not isinstance(entry, dict):
            raise TypeError("Each recovered-code entry must be an object")
        code = {key: copy.deepcopy(value) for key, value in entry.items() if key in CODE_FIELDS}
        draft["code"].append(code)
        for key, value in code.items():
            if key != "content":
                findings.extend(_scan_operations(step_id, value, ["code", index, key]))
        if isinstance(code.get("content"), str):
            findings.extend(
                _code_findings(
                    step_id, f"/code/{index}/content", code["content"], code.get("script_path")
                )
            )
        elif code.get("content") is not None:
            raise TypeError("Recovered code content must be text or None")
    findings = list({f["finding_id"]: f for f in findings}.values())
    by_id = {f["finding_id"]: f for f in findings}
    applied, addressed, changes, rejected = [], set(), defaultdict(list), False
    for decision in decisions:
        if not isinstance(decision, ContentDecision):
            raise TypeError("Only typed ContentDecision objects are accepted")
        if decision.finding_id not in by_id:
            raise ValueError("Unknown or stale finding ID in manual decision")
        if decision.finding_id in addressed:
            raise ValueError("A finding may receive only one decision per compilation")
        finding = by_id[decision.finding_id]
        _validate_decision(decision, finding)
        current = _lookup(draft, finding["path"])
        if decision.action == "redact":
            if (
                not isinstance(current, str)
                or type(decision.start) is not int
                or type(decision.end) is not int
            ):
                raise ValueError("Redaction requires a string and integer character offsets")
            if (
                not 0 <= decision.start < decision.end <= len(current)
                or text_digest(current[decision.start : decision.end]) != decision.span_sha256
            ):
                raise ValueError("Redaction span bounds/hash do not match exact source")
            changes[finding["path"]].append(decision)
        elif decision.action == "remove_field":
            changes[finding["path"]].append(decision)
        elif decision.action == "reject":
            rejected = True
        addressed.add(decision.finding_id)
        applied.append(asdict(decision))
    modified = set()
    paths = sorted(changes)
    if any(b.startswith(a + "/") for a, b in pairwise(paths)):
        raise ValueError("Manual actions cannot modify nested overlapping fields")
    for path, actions in changes.items():
        original = _lookup(draft, path)
        if any(d.action == "remove_field" for d in actions):
            if len(actions) != 1:
                raise ValueError("Whole-field removal cannot overlap another manual action")
            _replace(draft, path, remove=True)
        else:
            ordered = sorted(actions, key=lambda d: d.start)
            if any(a.end > b.start for a, b in pairwise(ordered)):
                raise ValueError("Manual redaction spans overlap")
            value = original
            for action in reversed(ordered):
                value = value[: action.start] + value[action.end :]
            _replace(draft, path, value)
        modified.add(path)
        removed.append(
            {
                "path": path,
                "kind": "manual_content_bound_change",
                "source_sha256": digest(original),
                "content": original,
                "spans": [_span(original, d.start, d.end) for d in actions if d.action == "redact"],
            }
        )
    unresolved = [
        f
        for f in findings
        if f["finding_id"] not in addressed
        and not any(f["path"] == p or f["path"].startswith(p + "/") for p in modified)
    ]
    for path in modified:
        try:
            value = _lookup(draft, path)
        except (KeyError, IndexError):
            continue
        if path.startswith("/code/") and path.endswith("/content"):
            code = draft["code"][int(_parts(path)[1])]
            unresolved.extend(_code_findings(step_id, path, value, code.get("script_path")))
        else:
            unresolved.extend(_scan_operations(step_id, value, _parts(path)))
    unresolved = list({f["finding_id"]: f for f in unresolved}.values())
    return {
        "step_id": step_id,
        "status": "rejected" if rejected else "needs_review" if unresolved else "draft_clear",
        "draft_input": draft,
        "agent_payload": None,
        "audit": {
            "schema": SCHEMA,
            "source_input_sha256": source_hash,
            "draft_input_sha256": digest(draft),
            "semantic_certificate": False,
            "requires_whole_step_content_review": True,
            "removed": removed,
            "findings": findings,
            "unresolved_findings": unresolved,
            "applied_decisions": applied,
            "finding_counts": dict(Counter(f["kind"] for f in unresolved)),
            "ignored_model_input_fields": sorted(set(source) - {"task", "plan", "code"}),
            "limits": [
                "Scanner absence is not evidence of semantic absence; implicit or paraphrased results may remain.",
                "Raw removed content and finding snippets are PRIVATE and must not enter agent payloads.",
                "Operational parameters and code are unchanged unless a content-bound trusted decision explicitly changes them.",
                "No target labels, prior observations, or known-previous-checkpoint fields outside model_input are accessed.",
            ],
        },
    }


def compile_steps(rows, *, decisions_by_step=None):
    """Compile an iterable in memory; no CLI, file writes, or external calls."""
    decisions_by_step = decisions_by_step or {}
    results = [
        compile_step(row, decisions=decisions_by_step.get(row["example_id"], ())) for row in rows
    ]
    ids = [result["step_id"] for result in results]
    if len(ids) != len(set(ids)):
        raise ValueError("Step IDs must be unique")
    if set(decisions_by_step) - set(ids):
        raise ValueError("Manual decisions include unknown step IDs")
    return results, {
        "schema": SCHEMA,
        "counts": dict(Counter(r["status"] for r in results)),
        "steps": len(results),
        "semantic_certificate": False,
    }
