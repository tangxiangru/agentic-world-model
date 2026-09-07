"""Bounded explicit reuse screening for an archived checkpoint's mutable source.

The caller validates raw files against its pinned download manifest. This helper
additionally checks the registry's ledger hash and records hashes for every raw
file it examines. ``clear`` means no explicit conflicting evidence was found,
not that arbitrary execution, symlink identity, or checkpoint bytes are certified.
No labels, tool-result prose, or predictor scores participate in screening.
"""

from __future__ import annotations

import ast
import hashlib
import json
import posixpath
import re
import shlex
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from tools.outcome_prediction.rpm_code_provenance import TraceIndex, _python_targets

OUTPUT_KEYS = {"output_dir", "output_path", "output_checkpoint", "save_dir", "save_path"}
OUTPUT_FLAGS = OUTPUT_KEYS | {"output", "out", "dest", "destination", "dst", "checkpoint_dir"}
MODEL_FILES = {
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "model.safetensors", "pytorch_model.bin", "adapter_config.json",
}
LIMITATIONS = [
    "Explicit declarations and recorded mutation attempts only; not checkpoint-byte or whole-program certification.",
    "Unknown/dynamic paths, indirect imported code, unlogged processes, and symlink targets are not resolved.",
    "The child itself is excluded from proposed-output conflicts because in-place initialization may be legitimate; no child execution guarantee is made.",
    "Caller must verify raw record/trace hashes against the pinned download manifest; newly computed hashes alone are not that verification.",
]


def _stamp(value):
    try:
        at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return at if at.tzinfo is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def _path(value, cwd="/home/ben/task"):
    if not isinstance(value, str) or not value.strip():
        return None
    if re.search(r"[$`{}*?<>\n\r]", value) or value.startswith("~"):
        return None
    if not isinstance(cwd, str) or not cwd.startswith("/"):
        return None
    return posixpath.normpath(posixpath.join(cwd, value.strip()))


def _overlaps(a, b):
    return a == b or a.startswith(b.rstrip("/") + "/") or b.startswith(a.rstrip("/") + "/")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _equal_second_order(parent, child):
    """Bind the child plan to the same verified ledger used by its parent."""
    evidence, provenance = parent.get("evidence") or {}, child.get("provenance") or {}
    try:
        path = Path(provenance["ledger_path"])
        if str(path) != evidence.get("ledger_path"):
            return None
        digest = _sha(path)
        if digest != provenance["ledger_sha256"] or digest != evidence.get("ledger_sha256"):
            return None
        first_name = Path(provenance["first_record_path"]).name
        ledger = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        found = [(i, e) for i, e in enumerate(ledger, 1) if isinstance(e, dict)
                 and e.get("card_id") == child.get("card_id") and e.get("event") == "submit"
                 and Path(e.get("path") or "").name == first_name]
        if (len(found) != 1 or found[0][1].get("stage") != "plan"
                or _stamp(found[0][1].get("at")) != _stamp(child.get("first_submitted_at"))
                or type(evidence.get("ledger_line")) is not int
                or evidence["ledger_line"] >= found[0][0]):
            return None
        return {"parent_line": evidence["ledger_line"], "child_line": found[0][0], "ledger": ledger}
    except (KeyError, OSError, ValueError, TypeError):
        return None


def _proposed_outputs(row):
    setup = ((row.get("model_input") or {}).get("plan") or {}).get("setup") or {}
    command = setup.get("command") or {}
    cwd = command.get("cwd") or "/home/ben/task"
    found = []
    for key in sorted(OUTPUT_KEYS):
        path = _path(setup.get(key), cwd)
        if path:
            found.append({"path": path, "locator": "first_plan.setup." + key})
    argv = command.get("argv")
    if isinstance(argv, str):
        try:
            argv = shlex.split(argv)
        except ValueError:
            argv = []
    if not isinstance(argv, list) or not all(isinstance(t, str) for t in argv):
        return found
    for i, token in enumerate(argv):
        flag, sep, value = token.partition("=")
        if not flag.startswith("--") or flag[2:].replace("-", "_") not in OUTPUT_FLAGS:
            continue
        value = value if sep else argv[i + 1] if i + 1 < len(argv) else None
        path = _path(value, cwd)
        if path:
            found.append({"path": path, "locator": "first_plan.setup.command.argv[" + str(i) + "]"})
    return found


def _model_path(path, source):
    """Whole-directory operations or recognized model/config files within it."""
    if path is None:
        return False
    if path == source or source.startswith(path.rstrip("/") + "/"):
        return True
    if not path.startswith(source.rstrip("/") + "/"):
        return False
    name = posixpath.basename(path)
    return name in MODEL_FILES or name.endswith((".safetensors", ".bin", ".pt", ".pth", ".index.json", ".model", ".jinja"))


def _shell_evidence(command, source, cwd):
    """Token-aware shell destinations plus nonexecuting literal Python writes."""
    if not isinstance(command, str):
        return []
    # Recognize an initial literal cd; unknown/dynamic chdir remains a limitation.
    initial = re.match(r"^\s*cd\s+([^;&|\n]+)\s*(?:&&|;)", command)
    if initial:
        try:
            tokens = shlex.split(initial.group(1))
        except ValueError:
            tokens = []
        if len(tokens) == 1:
            cwd = _path(tokens[0], cwd) or cwd
    candidates = {source, *(posixpath.join(source, name) for name in MODEL_FILES)}
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    for token in tokens:
        path = _path(token, cwd)
        if _model_path(path, source):
            candidates.add(path)
    # String constants are candidates, never interpreted as code to execute.
    snippets, shell_command = [], command
    for match in reversed(list(re.finditer(r"([^\n]*)<<-?\s*(['\"]?)(\w+)\2[^\n]*\n(.*?)\n\3(?:\s|$)", command, re.DOTALL))):
        if re.search(r"\bpython(?:3(?:\.\d+)?)?\b", match.group(1)):
            snippets.append(match.group(4))
        shell_command = shell_command[:match.start()] + match.group(1) + shell_command[match.end():]
    operations = []

    def destination(value, reason):
        path = _path(value, cwd)
        if _model_path(path, source):
            operations.append({"path": path, "operations": [reason]})

    for line in shell_command.splitlines():
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
            lexer.whitespace_split = True
            segments = [[]]
            for token in lexer:
                if token and all(c in ";&|" for c in token):
                    segments.append([])
                else:
                    segments[-1].append(token)
        except ValueError:
            continue
        for segment in segments:
            if not segment:
                continue
            program = posixpath.basename(segment[0])
            if program == "cd" and len(segment) == 2:
                cwd = _path(segment[1], cwd) or cwd
                continue
            for i, token in enumerate(segment[:-1]):
                if token in {">", ">>", "&>"}:
                    destination(segment[i + 1], "shell output redirection")
            if re.fullmatch(r"python(?:3(?:\.\d+)?)?", program):
                for i, token in enumerate(segment[:-1]):
                    if token == "-c":
                        snippets.append(segment[i + 1])
            args = [t for t in segment[1:] if not t.startswith("-")]
            if program in {"rm", "mv", "truncate", "tee"}:
                for value in args:
                    destination(value, "shell " + program + " mutation")
            elif program in {"cp", "install", "ln"} and args:
                destination(args[-1], "shell " + program + " destination")
                for i, token in enumerate(segment[:-1]):
                    if token in {"-t", "--target-directory"}:
                        destination(segment[i + 1], "shell " + program + " target directory")
            elif program in {"sed", "perl"} and any(re.match(r"-\w*i", t) for t in segment[1:]):
                for value in args:
                    destination(value, "shell in-place " + program)
    for snippet in snippets:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(snippet)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                path = _path(node.value, cwd)
                if _model_path(path, source):
                    candidates.add(path)
    for snippet in snippets:
        for path in sorted(candidates):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                found = _python_targets(snippet, path, cwd)
            if found:
                operations.append({"path": path, "operations": found})
    return operations


@lru_cache(maxsize=8)
def _trace(path, digest):
    index = TraceIndex(path)
    if index.trace_sha256 != digest:
        raise ValueError("Trace changed during mutation audit")
    return index


def audit_source_use(parent_registry_entry, child_row, all_rows, registry):
    """Audit a parent's injected ``consumed_path`` without consulting labels.

    ``parent_registry_entry`` is a registry copy augmented by the caller with
    the exact resolved consumed_path. Registry source_path/archive_path, archive
    time, and evidence ledger/record paths are assumed to have passed binding.
    """
    parent = parent_registry_entry
    result = {"status": "unresolved", "reasons": [], "evidence": [], "limitations": list(LIMITATIONS)}

    def hold(reason, **evidence):
        result["reasons"].append(reason)
        if evidence:
            result["evidence"].append({"kind": reason, **evidence})

    def finish():
        result["reasons"] = sorted(set(result["reasons"]))
        result["status"] = "unresolved" if result["reasons"] else "clear"
        return result

    start, cutoff = _stamp(parent.get("archived_at")), _stamp(child_row.get("first_submitted_at"))
    source, archive, consumed = (_path(parent.get(k)) for k in ("source_path", "archive_path", "consumed_path"))
    if (
        start is None or cutoff is None or start > cutoff
        or parent.get("cell_id") != child_row.get("cell_id")
        or parent.get("example_id") == child_row.get("example_id")
        or not source or not archive or consumed not in {source, archive}
    ):
        hold("invalid_source_use_context")
        return finish()
    equal_order = _equal_second_order(parent, child_row) if start == cutoff else None
    if start == cutoff and equal_order is None:
        hold("equal_timestamp_order_unproven")
        return finish()
    result["evidence"].append({"kind": "interval", "after_archive": parent["archived_at"], "before_proposal": child_row["first_submitted_at"], "consumed_path": consumed})
    if equal_order:
        result["evidence"].append({"kind": "equal_second_ledger_order", "parent_line": equal_order["parent_line"], "child_line": equal_order["child_line"]})
    if consumed == archive:
        result["evidence"].append({"kind": "immutable_archive_path", "path": archive})
        return finish()

    same_cell = [r for r in all_rows if r.get("cell_id") == child_row.get("cell_id")]
    ids = [r.get("example_id") for r in same_cell]
    if len(set(ids)) != len(ids) or parent.get("example_id") not in ids:
        hold("incomplete_or_duplicate_mutation_inventory")
        return finish()
    for other in same_cell:
        oid = other.get("example_id")
        if oid in {parent.get("example_id"), child_row.get("example_id")}:
            continue
        when = _stamp(other.get("first_submitted_at"))
        for output in _proposed_outputs(other):
            if not _overlaps(output["path"], source):
                continue
            if when is None:
                hold("overlapping_proposal_time_unknown", example_id=oid, **output)
            elif start <= when < cutoff:
                hold("other_card_proposes_overlapping_output", example_id=oid, proposed_at=other["first_submitted_at"], **output)
            elif equal_order and when == start:
                event_lines = [i for i, e in enumerate(equal_order["ledger"], 1)
                               if e.get("card_id") == other.get("card_id", oid.rsplit("/", 1)[-1])
                               and e.get("event") == "submit" and e.get("stage") == "plan"]
                if not event_lines or equal_order["parent_line"] < min(event_lines) < equal_order["child_line"]:
                    hold("overlapping_proposal_in_equal_second_interval", example_id=oid, ledger_lines=event_lines, **output)
        # A writer proposed earlier may finish inside the relevant interval.
        entry = registry.get(oid) or {}
        when = _stamp(entry.get("archived_at"))
        path = _path(entry.get("source_path"))
        in_interval = when is not None and start <= when < cutoff
        if equal_order and when == start:
            line = (entry.get("evidence") or {}).get("ledger_line")
            in_interval = type(line) is not int or equal_order["parent_line"] < line < equal_order["child_line"]
        if in_interval and path and _overlaps(path, source):
            hold("other_card_archives_overlapping_output", example_id=oid, archived_at=entry["archived_at"], path=path)

    evidence = parent.get("evidence") or {}
    ledger_name, record_name = evidence.get("ledger_path"), evidence.get("record_path")
    if not ledger_name or not record_name or not evidence.get("ledger_sha256"):
        hold("missing_mutation_ledger_provenance")
        return finish()
    ledger_path, record_path = Path(ledger_name), Path(record_name)
    try:
        if _sha(ledger_path) != evidence["ledger_sha256"]:
            hold("mutation_ledger_hash_changed")
            return finish()
        ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        hold("unreadable_mutation_ledger")
        return finish()
    result["evidence"].append({"kind": "ledger_checked", "path": str(ledger_path), "sha256": evidence["ledger_sha256"]})
    for line, event in enumerate(ledger, 1):
        if not isinstance(event, dict) or event.get("event") != "submit" or event.get("card_id") != parent.get("card_id"):
            continue
        when, number = _stamp(event.get("at")), event.get("record_n")
        if number == evidence.get("record_n"):
            continue
        if when is None:
            hold("same_card_submission_time_unknown", ledger_line=line)
            continue
        in_interval = start <= when < cutoff or bool(equal_order and when == start and equal_order["parent_line"] < line < equal_order["child_line"])
        if not in_interval:
            continue
        if type(number) is not int or number < 1:
            hold("invalid_intervening_record_number", ledger_line=line)
            continue
        path = record_path.parent / f"record-{number:02d}.json"
        try:
            record = json.loads(path.read_text())
            card = record.get("card") or {}
            at = _stamp(record.get("at"))
            if card.get("card_id") != parent.get("card_id") or at is None or at > when:
                raise ValueError("Record identity/time mismatch")
        except (OSError, ValueError, TypeError, AttributeError):
            hold("unreadable_or_mismatched_intervening_record", path=str(path), ledger_line=line)
            continue
        card_result = card.get("result") or {}
        output = _path(card_result.get("output_checkpoint")) if isinstance(card_result, dict) else None
        changed = card.get("setup") != parent.get("archived_setup") or output != source
        item = {"kind": "intervening_same_card_record", "path": str(path), "sha256": _sha(path), "ledger_line": line, "at": event["at"], "setup_changed": card.get("setup") != parent.get("archived_setup"), "output_path": output}
        result["evidence"].append(item)
        if changed:
            hold("same_card_setup_or_output_changed_after_archive")

    trace_path = Path(evidence.get("trace_path") or ledger_path.parent.parent / "solve_out_sanitized.txt")
    if not trace_path.is_file():
        hold("mutation_trace_unavailable", path=str(trace_path))
        return finish()
    try:
        digest = _sha(trace_path)
        if evidence.get("trace_sha256") and evidence["trace_sha256"] != digest:
            hold("mutation_trace_hash_changed")
            return finish()
        index = _trace(str(trace_path), digest)
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        hold("unreadable_mutation_trace")
        return finish()
    result["evidence"].append({"kind": "trace_checked", "path": str(trace_path), "sha256": digest, "malformed_json_events": index.malformed_json_events})
    if index.malformed_json_events:
        hold("malformed_trace_events")
    cwd = archive.rsplit("/wm/checkpoints/", 1)[0] if "/wm/checkpoints/" in archive else "/home/ben/task"
    for call in index.calls:
        # A write launched before the archive but completed afterwards overlaps.
        end = (call.get("result") or {}).get("time") or call["time"]
        if (call["time"] >= cutoff or end < start) and not (equal_order and call["time"] <= cutoff <= end):
            continue
        inp = call.get("input") or {}
        if not isinstance(inp, dict):
            hold("malformed_intervening_tool_input", trace_line=call["line"])
            continue
        changes = []
        if call.get("name") in {"Write", "Edit", "MultiEdit"}:
            path = _path(inp.get("file_path"), cwd)
            if _model_path(path, source):
                changes.append({"path": path, "operations": ["recorded " + call["name"] + " attempt"]})
        elif call.get("name") == "Bash":
            changes = _shell_evidence(inp.get("command"), source, cwd)
        if changes:
            hold("explicit_checkpoint_mutation_attempt", trace_line=call["line"], tool_id=call["id"], at=call["at"], changes=changes)
    return finish()
