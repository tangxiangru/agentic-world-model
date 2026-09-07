"""Clean-only recipe features and conservative exact-path checkpoint context.

Official labels enter only through the frozen clean train/test loaders. Private
inventory label objects and prior measurement objects are lexically skipped.
An immediate-parent score needs one distinct consumed checkpoint, a unique
clean producer, and an earlier closed registration naming its scored output.
This establishes declared path/timing consistency, not executed byte identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from tools.outcome_prediction.wm_clean_refresh import load_partition
from tools.outcome_prediction.wm_code_features import _command_tokens, _script_index
from tools.outcome_prediction.wm_parameter_pilot import _value_end
from tools.outcome_prediction.wm_small_data import step_features

WEIGHT_FLAGS = {
    "model",
    "model_path",
    "model_name_or_path",
    "model_name",
    "weights",
    "init",
    "init_from",
    "init_model",
    "base",
    "base_model",
    "checkpoint",
    "ckpt",
    "resume_from_checkpoint",
    "src",
}
MERGE_FLAGS = {"models", "srcs", "inputs", "a", "b", "sources"}


def _fields(raw):
    """Return raw field slices, never decoding excluded values."""
    raw = raw.strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        raise ValueError("Expected JSON object")
    cursor, result, decoder = 1, {}, json.JSONDecoder()
    while cursor < len(raw) - 1:
        while raw[cursor].isspace():
            cursor += 1
        if raw[cursor] == "}":
            break
        key, cursor = decoder.raw_decode(raw, cursor)
        if not isinstance(key, str) or key in result:
            raise ValueError("Duplicate or invalid JSON key")
        while raw[cursor].isspace():
            cursor += 1
        if raw[cursor] != ":":
            raise ValueError("Missing JSON field separator")
        cursor += 1
        while raw[cursor].isspace():
            cursor += 1
        end = _value_end(raw, cursor)
        result[key] = raw[cursor:end].strip()
        cursor = end
        if raw[cursor] == ",":
            cursor += 1
            if not raw[cursor:-1].strip():
                raise ValueError("Trailing JSON comma")
        elif raw[cursor] != "}":
            raise ValueError("Malformed JSON object")
    return result


def _array(raw):
    raw = raw.strip()
    if not raw.startswith("[") or not raw.endswith("]"):
        raise ValueError("Expected JSON array")
    cursor = 1
    while cursor < len(raw) - 1:
        while raw[cursor].isspace():
            cursor += 1
        if raw[cursor] == "]":
            break
        end = _value_end(raw, cursor)
        yield raw[cursor:end].strip()
        cursor = end
        if raw[cursor] == ",":
            cursor += 1
        elif raw[cursor] != "]":
            raise ValueError("Malformed JSON array")


def _object(fields, key):
    raw = fields.get(key, "{}")
    return _fields(raw) if raw.startswith("{") else {}


def _take(fields, key, default=None):
    return json.loads(fields[key]) if key in fields else default


def _stamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.tzinfo is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _path(value, cwd):
    if not isinstance(value, str) or not value.strip():
        return None
    if re.search(r"[$`{}*?\n\r]", value) or value.startswith("~"):
        return None
    value = value.strip()
    if value.startswith("/"):
        return posixpath.normpath(value)
    if not isinstance(cwd, str) or not cwd.startswith("/"):
        return None
    return posixpath.normpath(posixpath.join(cwd, value))


def _base_identity(value, base):
    if not isinstance(value, str) or not isinstance(base, str) or not base:
        return False
    if value == base:
        return True
    return bool(
        re.search(
            r"(?:^|/)models--" + re.escape(base.replace("/", "--")) + r"/snapshots/[0-9a-f]{40}/?$",
            value,
        )
    )


def _project_line(line, clean_ids):
    top = _fields(line)
    result = {
        key: _take(top, key)
        for key in (
            "example_id",
            "cell_id",
            "card_id",
            "benchmark",
            "first_submitted_at",
            "first_stage",
        )
    }
    source = _object(top, "model_input")
    plan = _object(source, "plan")
    setup = _object(plan, "setup")
    command = _object(setup, "command")
    result["command"] = {k: _take(command, k) for k in ("argv", "script", "cwd") if k in command}
    parent = _object(setup, "parent_checkpoint")
    result["parent_path"] = _take(parent, "path")
    task = _object(source, "task")
    result["base_model"] = _take(setup, "base_model") or _take(task, "base_model")
    method = _object(setup, "method")
    result["family"] = _take(method, "family", "")
    result["declared_output"] = _take(setup, "output_dir")
    audit = _object(top, "audit")
    binding = _object(audit, "output_artifact_comparison")
    result["scored_output"] = _take(binding, "final_output_checkpoint")
    result["provenance"] = {
        k: _take(_object(top, "provenance"), k)
        for k in ("first_record_sha256", "card_sha256", "official_metric_sha256")
    }
    # Only retained targets supply model features. Neither excluded-row code
    # nor excluded-row numeric hyperparameters are decoded.
    if result["example_id"] in clean_ids:
        retained_setup = {k: _take(setup, k) for k in ("method", "data", "budget") if k in setup}
        result["step_row"] = {
            "first_stage": result["first_stage"],
            "audit": {"reasons": []},
            "model_input": {
                "plan": {"setup": retained_setup},
                "code": _take(source, "code", []),
            },
        }
        closed = []
        for raw in _array(top.get("prior_observations", "[]")):
            prior = _fields(raw)
            if _take(prior, "stage") != "closed":
                continue
            identifier = _take(prior, "example_id")
            if identifier not in clean_ids:
                continue
            record = _object(prior, "record")
            card = _object(record, "card")
            output = _object(card, "result")
            closed.append(
                {
                    "example_id": identifier,
                    "at": _take(prior, "at"),
                    "record_at": _take(record, "at"),
                    "record_card_id": _take(card, "card_id"),
                    "output_checkpoint": _take(output, "output_checkpoint"),
                    "record_sha256": _take(_object(prior, "provenance"), "sha256"),
                }
            )
        result["earlier_closed"] = closed
    return result


def _consumed(row):
    command = row["command"]
    argv = _command_tokens(command)
    start = _script_index(argv)
    if start is None:
        return [], "unsupported_or_ambiguous_command"
    cwd = command.get("cwd")
    declared = command.get("script")
    if declared and _path(declared, cwd) != _path(argv[start], cwd):
        return [], "command_script_conflict"
    merge = bool(re.search(r"merge|soup|averag", str(row["family"]), re.IGNORECASE)) or bool(
        re.search(r"merge|soup|averag", posixpath.basename(argv[start]), re.IGNORECASE)
    )
    flags = WEIGHT_FLAGS | (MERGE_FLAGS if merge else set())
    found, index, saw_flag = [], start + 1, False
    while index < len(argv):
        token = argv[index]
        if token == "--":
            break
        index += 1
        if not token.startswith("--"):
            continue
        flag, sep, inline = token[2:].partition("=")
        flag = flag.replace("-", "_")
        if flag not in flags:
            continue
        saw_flag = True
        values = [inline] if sep else []
        if not sep:
            while index < len(argv) and not argv[index].startswith("-"):
                values.append(argv[index])
                index += 1
        if not values:
            return [], "weight_argument_missing_value"
        if flag not in MERGE_FLAGS and len(values) != 1:
            return [], "ambiguous_weight_argument"
        for value in values:
            for raw in value.split(",") if merge else [value]:
                if merge:
                    match = re.fullmatch(r"(.+):[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw)
                    raw = match[1] if match else raw
                found.append(raw)
    if not saw_flag:
        if not isinstance(row["parent_path"], str) or not row["parent_path"]:
            return [], "no_explicit_checkpoint_input"
        if merge:
            return [], "merge_inputs_not_explicit"
        found = [row["parent_path"]]
    references = []
    for raw in found:
        base = _base_identity(raw, row["base_model"])
        path = None if base else _path(raw, cwd)
        references.append({"raw": raw, "is_base": base, "path": path})
    unique = {
        ("base" if r["is_base"] else "path", r["raw"] if r["is_base"] else r["path"]): r
        for r in references
    }
    return list(
        unique.values()
    ), "command_weight_argument" if saw_flag else "declared_parent_path_no_command_override"


def _closed_match(parent, target, path):
    cutoff, proposed = _stamp(target["first_submitted_at"]), _stamp(parent["first_submitted_at"])
    if cutoff is None or proposed is None or proposed >= cutoff:
        return None
    for event in target["earlier_closed"]:
        closed, recorded = _stamp(event["at"]), _stamp(event["record_at"])
        if (
            event["example_id"] == parent["example_id"]
            and event["record_card_id"] == parent["card_id"]
            and closed is not None
            and closed == recorded
            and proposed <= closed < cutoff
            and _path(event["output_checkpoint"], parent["command"].get("cwd")) == path
            and isinstance(event["record_sha256"], str)
            and len(event["record_sha256"]) == 64
        ):
            return event
    return None


def _features(current, selected, reference, known, base, view):
    result = {
        "parent_reference": reference,
        "reference_known": float(known),
        "input.base_declared": None if base is None else float(base),
        **{"current." + k: v for k, v in current.items()},
    }
    if view == "current":
        return result
    result["history.count"] = float(len(selected))
    for key in sorted(set(current).union(*(set(s) for s in selected))):
        values = [(i, s.get(key)) for i, s in enumerate(selected) if s.get(key) is not None]
        latest = selected[-1].get(key) if selected else None
        result["history.mean." + key] = sum(v for _, v in values) / len(values) if values else None
        weights = [0.5 ** (len(selected) - 1 - i) for i, _ in values]
        result["history.decayed." + key] = (
            sum(w * item[1] for w, item in zip(weights, values)) / sum(weights) if values else None
        )
        result["history.latest." + key] = latest
        result["action_minus_latest." + key] = (
            current[key] - latest if current.get(key) is not None and latest is not None else None
        )
    return result


def build_context(bundle):
    """Return all clean targets, clean official labels, and a separate audit.

    The frozen holdout split is preserved. Unknown/base/multiple-parent
    references are centered on zero with ``reference_known=False``; zero is not
    an observed base/parent accuracy. Only a single resolved immediate parent
    supplies a score. Recursive history contains clean exact-path recipes only.
    """
    bundle = Path(bundle)
    clean, labels, partition = [], {}, {}
    for name in ("train", "test"):
        examples, scores = load_partition(bundle, name)
        clean.extend(examples)
        labels.update(scores)
        partition.update({r["example_id"]: name for r in examples})
    clean_ids = set(labels)
    if len(clean_ids) != len(clean) or {r["example_id"] for r in clean} != clean_ids:
        raise ValueError("Clean target identity mismatch")
    inventory_path = bundle / "private/inventory.jsonl"
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (
        manifest.get("private/inventory.jsonl")
        != hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Private inventory is not frozen by the clean bundle")
    metadata = {}
    for line in inventory_path.read_text().splitlines():
        if line.strip():
            row = _project_line(line, clean_ids)
            key = row["example_id"]
            if key in metadata or key != row["cell_id"] + "/" + row["card_id"]:
                raise ValueError("Invalid inventory identity")
            metadata[key] = row
    if not clean_ids <= metadata.keys():
        raise ValueError("Clean target missing from inventory")
    for row in clean:
        original = metadata[row["example_id"]]
        if any(row[k] != original[k] for k in ("cell_id", "benchmark")):
            raise ValueError("Clean/inventory metadata mismatch")
    owners = defaultdict(set)
    for row in metadata.values():
        for raw in (row["declared_output"], row["scored_output"]):
            path = _path(raw, row["command"].get("cwd"))
            if path:
                owners[(row["cell_id"], row["benchmark"], path)].add(row["example_id"])
    joined, context_audit = {}, {}
    for identifier in clean_ids:
        row = metadata[identifier]
        cutoff = _stamp(row["first_submitted_at"])
        references, source = _consumed(row)
        parents, edges = [], []
        for reference in references:
            path = reference["path"]
            edge = {
                **reference,
                "producer_id": None,
                "status": "base_unknown_accuracy" if reference["is_base"] else "unresolved",
            }
            if not reference["is_base"] and path:
                candidates = sorted(
                    owner
                    for owner in owners[(row["cell_id"], row["benchmark"], path)]
                    if owner != identifier
                    and (
                        _stamp(metadata[owner]["first_submitted_at"]) is None
                        or cutoff is not None
                        and _stamp(metadata[owner]["first_submitted_at"]) < cutoff
                    )
                )
                if len(candidates) != 1:
                    edge["status"] = "missing_or_ambiguous_exact_output"
                elif candidates[0] not in clean_ids:
                    edge["status"] = "producer_not_clean_labeled"
                else:
                    producer = metadata[candidates[0]]
                    scored = _path(producer["scored_output"], producer["command"].get("cwd"))
                    event = _closed_match(producer, row, path) if scored == path else None
                    if event is None:
                        edge["status"] = "no_matching_earlier_closed_output"
                    elif partition[producer["example_id"]] != partition[identifier]:
                        raise ValueError("Same-session parent crosses clean partition")
                    else:
                        edge.update(
                            producer_id=producer["example_id"],
                            status="clean_exact_path_earlier_closed",
                            closed_record=event,
                        )
                        parents.append(producer["example_id"])
            edges.append(edge)
        parents = sorted(set(parents), key=lambda p: (_stamp(metadata[p]["first_submitted_at"]), p))
        joined[identifier] = parents
        context_audit[identifier] = {
            "input_source": source,
            "references": edges,
            "from_base": all(r["is_base"] for r in references) if references else None,
            "reference_known": len(references) == 1 and len(parents) == 1,
            "immediate_inputs_resolved": bool(references)
            and all(r["is_base"] or r["producer_id"] is not None for r in edges),
            "runtime_or_checkpoint_bytes_certified": False,
        }

    def ancestors(identifier, active=frozenset()):
        if identifier in active:
            raise ValueError("Cycle in exact-path clean ancestry")
        found = set()
        for parent in joined[identifier]:
            found.add(parent)
            found.update(ancestors(parent, active | {identifier}))
        return sorted(found, key=lambda p: (_stamp(metadata[p]["first_submitted_at"]), p))

    steps = {key: step_features(metadata[key]["step_row"]) for key in clean_ids}
    output = []
    for row in clean:
        identifier = row["example_id"]
        audit = context_audit[identifier]
        parent_ids, history_ids = joined[identifier], ancestors(identifier)
        known, base = audit["reference_known"], audit["from_base"]
        reference = labels[parent_ids[0]] if known else 0.0
        views = {
            view: _features(
                steps[identifier], [steps[p] for p in chosen], reference, known, base, view
            )
            for view, chosen in (("current", []), ("parent", parent_ids), ("history", history_ids))
        }
        for features in views.values():
            if any(
                v is not None and (not isinstance(v, (float, int)) or not math.isfinite(v))
                for v in features.values()
            ):
                raise ValueError("Invalid numeric context feature")
        output.append(
            {
                **row,
                "partition": partition[identifier],
                "split": partition[identifier],
                "parent_reference": reference,
                "reference_known": known,
                "from_base": base,
                "parent_ids": parent_ids,
                "history_ids": history_ids,
                "views": views,
                "audit": {
                    **audit,
                    "history_scores_used": False,
                    "history_complete": audit["immediate_inputs_resolved"]
                    and all(context_audit[p]["immediate_inputs_resolved"] for p in history_ids),
                },
            }
        )
    audit = {
        "schema": "wm-clean-exact-parent-context-v1",
        "clean_bundle_manifest_sha256": hashlib.sha256(
            (bundle / "manifest.json").read_bytes()
        ).hexdigest(),
        "inventory_sha256": manifest["private/inventory.jsonl"],
        "targets": len(output),
        "partitions": dict(Counter(r["partition"] for r in output)),
        "reference_known": sum(r["reference_known"] for r in output),
        "base_declared": sum(r["from_base"] is True for r in output),
        "unknown_or_multiple_parent_reference": sum(
            not r["reference_known"] and r["from_base"] is not True for r in output
        ),
        "history_recipes": sum(len(r["history_ids"]) for r in output),
        "by_benchmark": {
            benchmark: {
                "targets": sum(r["benchmark"] == benchmark for r in output),
                "reference_known": sum(
                    r["benchmark"] == benchmark and r["reference_known"] for r in output
                ),
                "base_declared": sum(
                    r["benchmark"] == benchmark and r["from_base"] is True for r in output
                ),
                "with_history": sum(
                    r["benchmark"] == benchmark and bool(r["history_ids"]) for r in output
                ),
            }
            for benchmark in sorted({r["benchmark"] for r in output})
        },
        "rows": context_audit,
        "limits": [
            "Only clean labels from load_partition are used; private label/measurement fields are not decoded.",
            "Zero reference with reference_known=False is mathematical centering, not observed base accuracy.",
            "Origin hints, checkpoint prefix aliases, dirty/unlabeled ancestors and generic ancestor scores do not supply context.",
            "Multiple distinct input checkpoints are not reduced to a fabricated average parent score.",
            "Exact declared paths and earlier closed registration are not executed-weight-byte or historical grade-availability certificates.",
            "Partial clean recipe history is explicitly marked incomplete; no target is dropped for missing context.",
        ],
    }
    return output, labels, audit
