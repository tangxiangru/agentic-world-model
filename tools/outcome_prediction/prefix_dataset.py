"""Extract checkpoint-indexed code prefixes, generation configs, and ten-run labels.

This is a retrospective extraction, not proof that a recipe was predeclared or
reproducible. Inputs contain code/commands, never tool outputs or card outcomes.
Raw evidence and incomplete/unverified state are reported separately. Run with:

    python -m tools.outcome_prediction.prefix_dataset --out <new-directory>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import shlex
import tokenize
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.outcome_prediction.rpm_code_provenance import (
    TraceIndex,
    explicit_shell_mutations,
    normalize_path,
    timestamp,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIRROR = ROOT / "data/traj/raw/awm-gsm8k-trajectories-cc2ac9d884a7"
SCHEMA = "trajectory-prefix-code-generation-v1"
QUARANTINE = {
    "r0-25-exp-02",
    "aime-r0-11-exp-02",
    "aime2-r0-12-exp-05",
}
CODE_SUFFIXES = {".py", ".sh", ".bash"}
OUTCOME_LITERAL = re.compile(
    r"(?i)(?:accuracy|acc10|pass[_ -]?rate|score|correct|baseline|eval_result)"
    r"[^\n]{0,50}(?:\b\d+\.\d+|\b\d+\s*%|\b\d+\s*/\s*\d+)"
)
PERCENT_LITERAL = re.compile(r"\b\d+(?:\.\d+)?\s*%")
MEASURED_LOG = re.compile(
    r"(?i)(?:exp[-_ ]?\d|final_model|checkpoint|baseline|accuracy|score|pass.rate)[^\n]{0,100}?(?:\b\d+\.\d+|\b\d+\s*%)"
)
EXCLUDED_PATH = re.compile(r"(?:^|/)(?:wm|memory|logs?|results?|eval_results)(?:/|$)")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def strip_python_comments(source):
    """Drop comments/docstrings only; leave executable code, including literals.

    A flag records likely outcome-bearing executable literals. We do not mutate
    arbitrary numeric constants or claim that this is a semantic leakage review.
    """
    try:
        tree = ast.parse(source)
        spans = []
        print_spans = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    item = body[0]
                    spans.append(
                        (
                            item.lineno,
                            item.col_offset,
                            item.end_lineno,
                            item.end_col_offset,
                            "" if isinstance(node, ast.Module) else "pass",
                        )
                    )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
                and not any(keyword.arg == "file" for keyword in node.keywords)
            ):
                for arg in node.args:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and MEASURED_LOG.search(arg.value)
                    ):
                        print_spans.append(
                            (arg.lineno, arg.col_offset, arg.end_lineno, arg.end_col_offset)
                        )
        tokens = []
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                token = tokenize.TokenInfo(token.type, "", token.start, token.end, token.line)
            elif token.type == tokenize.STRING and any(
                (a, b) <= token.start and token.end <= (c, d) for a, b, c, d, _ in spans
            ):
                # pass keeps a docstring-only function/class syntactically valid.
                replacement = next(
                    value
                    for a, b, c, d, value in spans
                    if (a, b) <= token.start and token.end <= (c, d)
                )
                token = tokenize.TokenInfo(
                    token.type, replacement, token.start, token.end, token.line
                )
            elif token.type == tokenize.STRING and any(
                (a, b) <= token.start and token.end <= (c, d) for a, b, c, d in print_spans
            ):
                token = tokenize.TokenInfo(
                    token.type,
                    repr("[metric-bearing log message removed]"),
                    token.start,
                    token.end,
                    token.line,
                )
            tokens.append(token)
        return tokenize.untokenize(tokens), None
    except (SyntaxError, tokenize.TokenError, IndentationError):
        return source, "source_not_parseable_as_python; comments_not_removed"


def code_payload(content, path, kind):
    warning = None
    if kind == "file_version" and Path(path).suffix == ".py":
        cleaned, warning = strip_python_comments(content)
    else:
        # Do not strip shell text naively: a '#' can be data inside a heredoc.
        cleaned = sanitize_shell(content)
    flags = []
    if warning:
        flags.append(warning)
    if OUTCOME_LITERAL.search(cleaned):
        flags.append("possible_outcome_literal_requires_content_review")
    if PERCENT_LITERAL.search(cleaned):
        flags.append("percent_literal_requires_content_review")
    if MEASURED_LOG.search(cleaned):
        flags.append("possible_measured_log_requires_content_review")
    return {"kind": kind, "path": path, "content": cleaned}, flags


def sanitize_shell(source):
    """Remove obvious score-bearing log strings, without altering recipe values.

    Echo redirects may create training data, so never rewrite those automatically.
    Ambiguous surviving literals are flagged by code_payload instead.
    """
    pattern = re.compile(r"\b(?:echo|printf)\s+(\"(?:\\.|[^\"\\])*\"|'[^']*')")

    def replace(match):
        line_end = source.find("\n", match.end())
        rest = source[match.end() : line_end if line_end >= 0 else len(source)]
        if ">" in rest.split(";")[0] or "|" in rest.split(";")[0]:
            return match.group()
        if (
            OUTCOME_LITERAL.search(match.group(1))
            or PERCENT_LITERAL.search(match.group(1))
            or MEASURED_LOG.search(match.group(1))
        ):
            return match.group().replace(match.group(1), "'[metric-bearing log message removed]'")
        return match.group()

    cleaned = pattern.sub(replace, source)
    # Python inside a shell heredoc is code too: remove its comments/docstrings
    # and measured diagnostic print strings by the same AST-aware rule.
    replacements = []
    heredoc = re.compile(r"(?m)^(?P<header>[^\n]*?)<<-?\s*(['\"]?)(?P<tag>[A-Za-z_]\w*)\2[^\n]*\n")
    for match in heredoc.finditer(cleaned):
        header = cleaned[match.start() : match.end()]
        if not re.search(r"\bpython(?:3)?\b|\b(?:cat|tee)\b[^\n]*\.py\b", header):
            continue
        ending = re.search(
            r"(?m)^" + re.escape(match.group("tag")) + r"\s*$", cleaned[match.end() :]
        )
        if ending is not None:
            stop = match.end() + ending.start()
            body, warning = strip_python_comments(cleaned[match.end() : stop])
            if not warning:
                replacements.append((match.end(), stop, body))
    for start, stop, body in reversed(replacements):
        cleaned = cleaned[:start] + body + cleaned[stop:]
    return cleaned


def literal_shell_sources(command, cwd):
    """Recover quoted cat/tee heredocs without executing or expanding shell code."""
    pattern = re.compile(
        r"(?m)^(?P<header>[^\n]*?)<<-?\s*(?P<q>['\"])(?P<tag>[A-Za-z_]\w*)(?P=q)[^\n]*\n"
    )
    found = []
    for match in pattern.finditer(command):
        header = command[match.start() : match.end()].rstrip("\n")
        ending = re.search(
            r"(?m)^" + re.escape(match.group("tag")) + r"\s*$", command[match.end() :]
        )
        if ending is None:
            continue
        body = command[match.end() : match.end() + ending.start()]
        # Quoted delimiter guarantees literal heredoc bytes. <<- strips leading
        # tabs; represent its actual saved body, not the indentation in the trace.
        if "<<-" in header:
            body = "".join(line.lstrip("\t") for line in body.splitlines(keepends=True))
        path_match = re.search(r"(?<![<>])>\s*([^\s;]+)", header)
        if path_match is None:
            path_match = re.search(r"\btee\s+(?:--\s+)?([^\s;<>]+)", header)
        if path_match is None or not re.search(r"\b(?:cat|tee)\b", header):
            continue
        try:
            tokens = shlex.split(path_match.group(1))
        except ValueError:
            continue
        if len(tokens) != 1 or any(char in tokens[0] for char in "$`*?{}"):
            continue
        path = normalize_path(tokens[0], cwd)
        if source_path(path):
            found.append((path, body))
    return found


def load_cards(session_dir):
    """Use historical immutable submissions, never today's mutable card.json."""
    ledger = jsonl(session_dir / "wm/records.jsonl")
    grouped = {}
    for entry in sorted(ledger, key=lambda item: item["seq"]):
        if entry.get("event") == "submit" and entry.get("path"):
            grouped.setdefault(entry["card_id"], []).append(entry)
    cards = []
    for card_id, entries in grouped.items():
        first = entries[0]
        plans = [entry for entry in entries if entry.get("stage") == "plan"]
        plan = plans[0] if plans else first
        archives = [entry for entry in entries if entry.get("archived")]
        # submit_card reuses an existing archive. Repeated 'archived' ledger
        # values are not new copies; bind to the first archive creation.
        archive = archives[0] if archives else None
        load = lambda entry, card_id=card_id: read_json(
            session_dir / "wm/cards" / card_id / Path(entry["path"]).name
        )["card"]
        cards.append(
            {
                "card_id": card_id,
                "first": first,
                "plan": plan,
                "plan_card": load(plan),
                "archive": archive,
                "archive_card": load(archive) if archive else None,
                "archive_submission_count": len(archives),
                "submissions": entries,
            }
        )
    return sorted(cards, key=lambda item: (timestamp(item["first"]["at"]), item["first"]["seq"]))


def declared_code_names(card):
    setup = card.get("setup") or {}
    command = setup.get("command") or {}
    paths = set()
    declarations = []
    if isinstance(command, dict) and command.get("script"):
        declarations.append(command["script"])
    for item in setup.get("data") or []:
        if isinstance(item, dict) and item.get("built_by"):
            declarations.append(item["built_by"])
    for value in declarations:
        if not isinstance(value, str) or value.strip().lower() in {"none", "null", "n/a"}:
            continue
        # built_by is sometimes free text: 'a.py + b.py' is two files,
        # not a filename containing a plus sign. Inline prose is not a path.
        names = re.findall(r"(?<![\w])(?:/|\.\.?/)?[\w.-]+(?:/[\w.-]+)*\.(?:py|sh|bash)\b", value)
        paths.update(names)
    return paths


def declared_paths(card, cwd):
    return {normalize_path(name, cwd) for name in declared_code_names(card)}


def declared_alias(index, card, path, events, cutoff):
    """Resolve shorthand only against a unique observed, explicitly invoked file.

    This is audit-only: it cannot introduce code into the prediction input.
    """
    basename = Path(path).name
    if basename not in declared_code_names(card):
        return None
    candidates = {
        event["path"]
        for event in events
        if event["path"] and Path(event["path"]).name == basename and event["time"] < cutoff
    }
    if len(candidates) != 1:
        return None
    resolved = next(iter(candidates))
    for call in getattr(index, "calls", []):
        result = call.get("result")
        if (
            call["name"] != "Bash"
            or call["time"] >= cutoff
            or not result
            or result.get("is_error")
            or result["time"] >= cutoff
        ):
            continue
        try:
            lexer = shlex.shlex(
                call["input"].get("command", ""), posix=True, punctuation_chars=";&|\n"
            )
            lexer.whitespace = " \t\r"
            lexer.whitespace_split = True
            chunks, chunk = [], []
            ambiguous_flow = False
            for token in lexer:
                if token and all(char in ";&|\n" for char in token):
                    ambiguous_flow |= any(char in token for char in "&|")
                    chunks.append(chunk)
                    chunk = []
                else:
                    chunk.append(token)
            chunks.append(chunk)
            if ambiguous_flow:
                continue
        except ValueError:
            continue
        cwd = index.cwd
        for chunk in chunks:
            if chunk[:1] == ["cd"]:
                cwd = normalize_path(chunk[1], cwd) if len(chunk) == 2 else None
                continue
            if cwd is None:
                continue
            while chunk and re.fullmatch(r"[A-Za-z_]\w*=.*", chunk[0]):
                chunk = chunk[1:]
            if (
                len(chunk) >= 2
                and re.fullmatch(r"(?:python(?:3)?|bash)", chunk[0])
                and not any(char in chunk[1] for char in "$`*?{}")
                and normalize_path(chunk[1], cwd) == resolved
            ):
                return {
                    "declared_path": path,
                    "resolved_path": resolved,
                    "tool_use_id": call["id"],
                    "call_line": call["line"],
                    "at": call["time"].isoformat(),
                    "reason": "unique observed basename with explicit pre-boundary invocation",
                }
    return None


def source_path(path):
    return Path(path).suffix in CODE_SUFFIXES and not EXCLUDED_PATH.search(path)


def administrative_card_source(content):
    """Recognize card-only serialization helpers, never general training files.

    A helper outside memory/cards can still contain completed result cards. Keep
    unknown/mixed-purpose code for review instead of deleting arbitrary numbers.
    """
    if not re.search(r"memory/cards|wm/cards|awm-experiment-card-v1", content):
        return False
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    simple = {
        "open",
        "print",
        "yaml.safe_load",
        "yaml.safe_dump",
        "json.load",
        "json.dump",
        "re.sub",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) in simple:
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "read",
            "write",
            "format",
            "replace",
        }:
            continue
        return False
    return True


def shell_is_recipe(command, paths):
    """Keep executable recipe commands; never include tool-result text.

    A static heuristic, not a proof of dependency. Exact retained and omitted
    call IDs are audited. Review flags remain visible outside the model input.
    """
    if not command or not isinstance(command, str):
        return False
    try:
        tokens = shlex.split(command, comments=True)
        if (
            len(tokens) == 3
            and re.fullmatch(r"python(?:\d(?:\.\d+)?)?", tokens[0])
            and tokens[1] == "-c"
        ):
            tree = ast.parse(tokens[2])
            if all(
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "print"
                and not node.value.keywords
                for node in tree.body
            ):
                return False
    except (ValueError, SyntaxError):
        pass
    # Commands that write cards/memos embed scores and rationales. The separately
    # indexed executable files preserve source even if the submission is mixed in.
    if re.search(
        r"(?:wm[ ._-]+(?:submit|outcome|record)\b|wm\.record|memory/cards|wm/cards|records\.jsonl)",
        command,
    ):
        return False
    if re.search(
        r"(?:\bcat\b|\btee\b|write_text|open\()[^\n]*(?:README|SKILL|\.md[\s'\"])", command
    ):
        return False
    if re.search(
        r"(?:cat|tee)\s*(?:>\s*)?[^\n]*\.(?:py|sh)\b[^\n]*<<|cat\s*<<[^\n]*>[^\n]*\.(?:py|sh)\b",
        command,
    ):
        return True
    if re.search(r"(?:^|\s)(?:python\d?(?:\.\d+)?|accelerate|torchrun|deepspeed|bash)\b", command):
        if any(Path(path).name in command for path in paths):
            return True
        if re.search(
            r"(?:load_dataset|save_pretrained|generation_config\.json|train|finetun|merge|soup|build|prepare|generate)",
            command,
            re.IGNORECASE,
        ):
            return True
    return bool(
        "generation_config.json" in command
        and re.search(r"(?:\bcp\b|\bmv\b|\bsed\b|\btee\b|(?<!<)>)", command)
    )


def heredoc_availability(call, path, cards, cwd):
    """Bound an initial literal write by a later recorder snapshot in that call.

    Ledger times have second resolution; use the end of that second, not its
    beginning. Otherwise retain the conservative tool-result timestamp.
    """
    result = call["result"]
    command = call["input"].get("command", "")
    first_line, _, body = command.partition("\n")
    header = re.fullmatch(r"cat\s+>\s*([^\s]+)\s*<<\s*(['\"])(\w+)\2\s*", first_line)
    if not header or normalize_path(header[1], cwd) != path:
        return result["time"], {}
    end = re.search(r"(?m)^" + re.escape(header[3]) + r"\s*$", body)
    if not end:
        return result["time"], {}
    remainder = body[end.end() :]
    for card in cards:
        for entry in card.get("submissions", []):
            if not any(normalize_path(p, cwd) == path for p in entry.get("snapshotted", [])):
                continue
            source = entry.get("source")
            if not isinstance(source, str):
                continue
            submits = re.finditer(r"(?m)^awm\s+wm\s+submit\s+([^\s]+)", remainder)
            submit = next((m for m in submits if normalize_path(m[1], cwd) == source), None)
            upper = timestamp(entry["at"]) + timedelta(seconds=1)
            if (
                submit is not None
                and call["time"] < upper < result["time"]
                and not explicit_shell_mutations(remainder[: submit.start()], path, cwd)
            ):
                return upper, {
                    "availability_basis": "literal write precedes same-call recorder snapshot",
                    "snapshot_attestation_seq": entry["seq"],
                    "snapshot_attestation_at": entry["at"],
                    "timestamp_precision_seconds": 1,
                }
    return result["time"], {}


def build_code_events(index, cards, session_dir=None):
    # Fixed policy: a future card must never change an earlier row's input.
    # Declarations are consulted only in the missing-code audit, not selection.
    declared = set()
    events, excluded = [], []
    for path, versions in index.snapshots.items():
        if not source_path(path):
            continue
        for version in versions:
            payload, flags = code_payload(version["content"], path, "file_version")
            events.append(
                {
                    "time": version["time"],
                    "line": version["result_line"],
                    "path": path,
                    "payload": payload,
                    "flags": flags,
                    "evidence": {
                        "tool_use_id": version["tool_use_id"],
                        "call_line": version["call_line"],
                        "result_line": version["result_line"],
                        "available_at": version["time"].isoformat(),
                        "source_content_sha256": hashlib.sha256(
                            version["content"].encode()
                        ).hexdigest(),
                    },
                }
            )
    for call in index.calls:
        if call["name"] != "Bash":
            continue
        command = call["input"].get("command", "")
        result = call.get("result")
        if result and not result.get("is_error"):
            for path, content in literal_shell_sources(command, index.cwd):
                payload, flags = code_payload(content, path, "file_version")
                available, attestation = heredoc_availability(call, path, cards, index.cwd)
                events.append(
                    {
                        "time": available,
                        "line": result["line"],
                        "path": path,
                        "payload": payload,
                        "flags": flags,
                        "evidence": {
                            "tool_use_id": call["id"],
                            "call_line": call["line"],
                            "result_line": result["line"],
                            "available_at": available.isoformat(),
                            "source_kind": "literal_quoted_shell_heredoc",
                            "source_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                            "execution_success_verified": False,
                            **attestation,
                        },
                    }
                )
        if not shell_is_recipe(command, declared):
            excluded.append(call["id"])
            continue
        payload, flags = code_payload(command, None, "shell_command")
        # A launch is not proof of successful completion, and its status is not a
        # feature. Both timestamps/status live in audit only, not in code payload.
        events.append(
            {
                "time": call["time"],
                "line": call["line"],
                "path": None,
                "payload": payload,
                "flags": flags,
                "evidence": {
                    "tool_use_id": call["id"],
                    "call_line": call["line"],
                    "available_at": call["time"].isoformat(),
                    "source_content_sha256": hashlib.sha256(command.encode()).hexdigest(),
                    "result_line": result["line"] if result else None,
                    "execution_success_verified": False,
                },
            }
        )
    if hasattr(index, "trace_path"):
        from tools.outcome_prediction.prefix_code_fragments import recover_fragment_versions
        from tools.outcome_prediction.prefix_code_recovery import recover_code_events

        recovered = recover_code_events(index, session_dir)["versions"]
        recovered.extend(recover_fragment_versions(index))
        for version in recovered:
            path, content = version["path"], version["content"]
            if not source_path(path) or any(
                event["path"] == path
                and event["time"] == version["time"]
                and event["evidence"].get("source_content_sha256") == version["sha256"]
                for event in events
            ):
                continue
            payload, flags = code_payload(content, path, "file_version")
            evidence = {
                key: value
                for key, value in version.items()
                if key not in {"path", "content", "time", "sha256"}
            }
            evidence.update(
                available_at=version["time"].isoformat(),
                source_content_sha256=version["sha256"],
            )
            events.append(
                {
                    "time": version["time"],
                    "line": version["result_line"],
                    "path": path,
                    "payload": payload,
                    "flags": flags,
                    "evidence": evidence,
                }
            )
    if session_dir is not None:
        for card in cards:
            snapshot_dir = Path(session_dir) / "wm/cards" / card["card_id"] / "snapshot"
            manifest_path = snapshot_dir / "MANIFEST.json"
            if not manifest_path.is_file():
                continue
            for item in read_json(manifest_path).get("files", []):
                relative = Path(item["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                path = normalize_path(str(relative), index.cwd)
                local = snapshot_dir / relative
                if not source_path(path) or not local.is_file():
                    continue
                observed = timestamp(item["at"])
                # Snapshots are overwritten by later submissions. Only source
                # versions strictly before the original archive are admissible.
                if card["archive"] and observed >= timestamp(card["archive"]["at"]):
                    continue
                raw = local.read_bytes()
                if hashlib.sha256(raw).hexdigest() != item.get("sha256") or len(raw) != item.get(
                    "bytes"
                ):
                    raise ValueError(f"Source snapshot hash mismatch: {local}")
                attestations = [
                    entry
                    for entry in card.get("submissions", [])
                    if item["path"] in entry.get("snapshotted", [])
                    and timestamp(entry["at"]) >= observed
                    and (
                        not card["archive"]
                        or timestamp(entry["at"]) <= timestamp(card["archive"]["at"])
                    )
                ]
                if not attestations:
                    continue
                if any(
                    event["path"] == path
                    and event["time"] <= observed
                    and event["evidence"].get("source_content_sha256") == item["sha256"]
                    for event in events
                ):
                    continue
                payload, flags = code_payload(raw.decode(), path, "file_version")
                events.append(
                    {
                        "time": observed,
                        "line": 0,
                        "path": path,
                        "payload": payload,
                        "flags": flags,
                        "evidence": {
                            "available_at": item["at"],
                            "source_kind": "hash_verified_pre_archive_snapshot",
                            "snapshot_path": str(local),
                            "source_content_sha256": item["sha256"],
                            "snapshot_attestation_seq": attestations[0]["seq"],
                            "snapshot_attestation_at": attestations[0]["at"],
                        },
                    }
                )
    retained, omitted_sources = [], []
    for event in events:
        if event["payload"]["kind"] == "file_version" and administrative_card_source(
            event["payload"]["content"]
        ):
            omitted_sources.append(
                {
                    "path": event["path"],
                    **event["evidence"],
                    "reason": "administrative result/plan card serialization helper, not recipe source",
                }
            )
        else:
            retained.append(event)
    index.omitted_source_events = omitted_sources
    return sorted(retained, key=lambda item: (item["time"], item["line"])), excluded


def recipe_prefix(index, cards, target, events):
    """Chronological prefix, including code before the first plan and later edits.

    All earlier experimental branches are included. Event groups use submission
    boundaries and do NOT claim that every event contributed to target weights.
    The flattened code list exactly equals the retained stream before cutoff.
    """
    archive = target["archive"]
    cutoff = timestamp(archive["at"])
    eligible = [card for card in cards if timestamp(card["first"]["at"]) < cutoff]
    if target not in eligible:
        eligible.append(target)
        eligible.sort(key=lambda item: (timestamp(item["first"]["at"]), item["first"]["seq"]))
    # Distribute events by first-submission boundaries, retaining preparation
    # before the first submission in the first recipe.
    groups = [{"exp_id": card["card_id"], "codes": []} for card in eligible]
    audits = [
        {"exp_id": card["card_id"], "first_submitted_at": card["first"]["at"], "code_evidence": []}
        for card in eligible
    ]
    boundaries = [timestamp(card["first"]["at"]) for card in eligible]
    for event in events:
        if event["time"] >= cutoff:
            continue
        group = 0
        for i, boundary in enumerate(boundaries):
            if boundary <= event["time"]:
                group = i
        groups[group]["codes"].append(event["payload"])
        audits[group]["code_evidence"].append({**event["evidence"], "flags": event["flags"]})
    missing = []
    for card in eligible:
        own_cutoff = min(cutoff, timestamp(card["archive"]["at"])) if card["archive"] else cutoff
        for path in declared_paths(card["plan_card"], index.cwd):
            if not any(event["path"] == path and event["time"] < own_cutoff for event in events):
                alias = declared_alias(index, card["plan_card"], path, events, own_cutoff)
                if alias:
                    next(item for item in audits if item["exp_id"] == card["card_id"]).setdefault(
                        "declared_path_aliases", []
                    ).append(alias)
                    continue
                missing.append(
                    {
                        "exp_id": card["card_id"],
                        "path": path,
                        "reason": "no complete source version observed or safely reconstructed before boundary",
                    }
                )
    return groups, audits, missing


def checkpoint_paths(card):
    final = card["archive_card"] or {}
    setup, result = final.get("setup") or {}, final.get("result") or {}
    # output_dir is often a parent/logging directory, not the archived model.
    values = [
        result.get("output_checkpoint") or setup.get("output_dir"),
        card["archive"].get("archived"),
    ]
    return [value for value in values if isinstance(value, str) and value]


def trajectory_inventory(mirror, selected=None):
    """Inventory all trajectories, including those without archived/ten-run targets."""
    inventory = []
    for directory in sorted((mirror / "cells").iterdir()):
        if not directory.is_dir() or (selected and directory.name not in selected):
            continue
        if not (directory / "solve_out_sanitized.txt").is_file():
            continue
        records = directory / "wm/records.jsonl"
        ledger = jsonl(records) if records.is_file() else []
        cards = sorted({entry["card_id"] for entry in ledger if entry.get("card_id")})
        archived = sorted({entry["card_id"] for entry in ledger if entry.get("archived")})
        inventory.append(
            {
                "trajectory_id": directory.name,
                "recorder_available": records.is_file(),
                "card_ids": cards,
                "recorded_archived_card_ids": archived,
                "unarchived_card_ids": sorted(set(cards) - set(archived)),
                "exp_ids": [],
                "status": "no_archived_checkpoint_in_manifest",
            }
        )
    return inventory


def evaluation_context(benchmark):
    # Pinned rescore10 eval kit; inherited config still may set a lower cap.
    return {
        "protocol": "rescore10",
        "n_runs": 10,
        "requested_max_tokens": 4000 if benchmark == "gsm8k" else 16000,
        "generation_config_mode": "auto",
        "backend": "vllm",
        "backend_version": "0.11.0",
        "chat_template": "gemma3.jinja" if benchmark == "gsm8k" else "qwen3.jinja",
        "effective_settings_verified": False,
    }


def unscored_trajectory(directory, inventory):
    """Preserve the full code prefix where no ten-run checkpoint target exists."""
    from tools.outcome_prediction.prefix_generation import index_session, resolve_for_checkpoint

    index = TraceIndex(directory / "solve_out_sanitized.txt")
    cards = load_cards(directory) if inventory["recorder_available"] else []
    events, omitted = build_code_events(index, cards, directory)
    times = [call["time"] for call in index.calls] + [event["time"] for event in events]
    cutoff = (max(times) + timedelta(microseconds=1)).isoformat() if times else None
    config = (
        resolve_for_checkpoint(index_session(directory, trace_index=index), ["final_model"], cutoff)
        if cutoff
        else {"generation_config": None, "status": "unknown"}
    )
    prompt = (directory / "prompt.txt").read_text() if (directory / "prompt.txt").is_file() else ""
    model = next(
        (name for name in ("google/gemma-3-4b-pt", "Qwen/Qwen3-4B-Base") if name in prompt), None
    )
    benchmark = "gsm8k" if model == "google/gemma-3-4b-pt" else "aime2025" if model else None
    return {
        "schema": SCHEMA,
        "trajectory_id": directory.name,
        "exp_id": None,
        "input": {
            "benchmark": benchmark,
            "base_model": model,
            "prefix_recipes": [{"exp_id": None, "codes": [event["payload"] for event in events]}],
            "generation_config": config["generation_config"],
        },
        "output": {
            "status": "missing",
            "n_runs": 0,
            "per_run_pass_rate": [],
            "avg_pass_rate": None,
        },
        "audit": {
            "trace_sha256": index.trace_sha256,
            "cutoff": cutoff,
            "boundary": "end_of_recorded_trajectory_not_archived_checkpoint",
            "omitted_source_events": index.omitted_source_events,
            "generation_config": config,
            "code_evidence": [event["evidence"] for event in events],
            "omitted_shell_call_ids": omitted,
            "checkpoint_binding": "no_verified_archived_checkpoint; final_model is task-requested path only",
        },
    }


def write_line(handle, obj):
    handle.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def build(mirror, out, metadata_dir=None, sessions=None):
    from tools.outcome_prediction.prefix_generation import index_session, resolve_for_checkpoint
    from tools.outcome_prediction.prefix_labels import load_label

    mirror, out = Path(mirror).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output directory: {out}")
    if out == mirror or mirror in out.parents:
        raise ValueError("Derived output must not be inside the read-only source mirror")
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            Path(__file__).with_name("prefix_generation.py"),
            Path(__file__).with_name("prefix_labels.py"),
            Path(__file__).with_name("prefix_code_recovery.py"),
            Path(__file__).with_name("prefix_code_fragments.py"),
            Path(__file__).with_name("rpm_code_provenance.py"),
        )
    }
    manifest_path = mirror / "rescore10/relay/gcs_manifest.json"
    checkpoints = read_json(manifest_path)["checkpoints"]
    if sessions:
        checkpoints = [item for item in checkpoints if item["hf_card"].split("/")[1] in sessions]
    by_session = {}
    for item in checkpoints:
        by_session.setdefault(item["hf_card"].split("/")[1], []).append(item)
    counters, problems = Counter(), []
    trajectory_rows = trajectory_inventory(mirror, sessions)
    receipts = {}
    if metadata_dir and (Path(metadata_dir) / "manifest.json").is_file():
        receipts = {
            item["exp_id"]: item
            for item in read_json(Path(metadata_dir) / "manifest.json").get("checkpoints", [])
        }
    files = {
        name: (out / f"{name}.jsonl").open("w")
        for name in ("inputs", "labels", "audit", "examples")
    }
    try:
        for session, targets in sorted(by_session.items()):
            directory = mirror / "cells" / session
            index = TraceIndex(directory / "solve_out_sanitized.txt")
            cards = load_cards(directory)
            by_card = {card["card_id"]: card for card in cards}
            code_events, omitted = build_code_events(index, cards, directory)
            generation = index_session(directory, trace_index=index)
            ids = []
            for checkpoint in sorted(targets, key=lambda item: item["id"]):
                exp_id = checkpoint["id"]
                ids.append(exp_id)
                counters["checkpoint_records"] += 1
                card_id = Path(checkpoint["hf_card"]).parent.name
                target = by_card.get(card_id)
                if target is None or target["archive"] is None:
                    problems.append({"exp_id": exp_id, "reason": "missing_archive_submission"})
                    counters["missing_archive_submission"] += 1
                    continue
                prefix, code_audit, missing = recipe_prefix(index, cards, target, code_events)
                cutoff = target["archive"]["at"]
                metadata = (
                    Path(metadata_dir) / exp_id / "generation_config.json" if metadata_dir else None
                )
                if metadata is not None and not metadata.is_file():
                    metadata = None
                receipt = (
                    receipts.get(exp_id, {}).get("files", {}).get("generation_config.json", {})
                )
                verified_metadata = bool(
                    metadata
                    and receipt.get("status") == "downloaded"
                    and receipt.get("source_uri")
                    == checkpoint["gs_path"].rstrip("/") + "/generation_config.json"
                    and receipt.get("sha256") == hashlib.sha256(metadata.read_bytes()).hexdigest()
                )
                config = resolve_for_checkpoint(
                    generation,
                    checkpoint_paths(target),
                    cutoff,
                    archived_config_path=metadata,
                    archived_config_verified=verified_metadata,
                )
                model_input = {
                    "benchmark": checkpoint["benchmark"],
                    "base_model": checkpoint["base_model"],
                    "prefix_recipes": prefix,
                    "generation_config": config["generation_config"],
                    "evaluation": evaluation_context(checkpoint["benchmark"]),
                }
                flags = sorted(
                    {
                        flag
                        for group in code_audit
                        for event in group["code_evidence"]
                        for flag in event["flags"]
                    }
                )
                quality = {
                    "generation_config_status": config["status"],
                    "missing_declared_code": bool(missing),
                    "content_review_required": bool(flags),
                    "multiple_archive_submissions": target["archive_submission_count"] > 1,
                    "quarantined": exp_id in QUARANTINE,
                    "fully_verified_prediction_input": False,
                }
                # Freeze the input before reading any score.
                frozen_hash = digest(model_input)
                input_record = {
                    "schema": SCHEMA,
                    "exp_id": exp_id,
                    "trajectory_id": session,
                    "input": model_input,
                    "input_sha256": frozen_hash,
                    "quality": quality,
                }
                write_line(files["inputs"], input_record)
                result_path = mirror / "rescore10/results" / f"{exp_id}.json"
                label = load_label(result_path, benchmark=checkpoint["benchmark"])
                counters["labels_" + label["status"]] += 1
                label_record = {"exp_id": exp_id, "trajectory_id": session, "output": label}
                write_line(files["labels"], label_record)
                write_line(files["examples"], {**input_record, "output": label})
                write_line(
                    files["audit"],
                    {
                        "exp_id": exp_id,
                        "archive_cutoff": cutoff,
                        "checkpoint_uri": checkpoint["gs_path"],
                        "original_checkpoint_paths": checkpoint_paths(target),
                        "archive_record": target["archive"],
                        "trace_path": str(index.trace_path),
                        "trace_sha256": index.trace_sha256,
                        "recipes": code_audit,
                        "missing_code": missing,
                        "generation_config": config,
                        "omitted_shell_call_ids": omitted,
                        "omitted_source_events": [
                            event
                            for event in index.omitted_source_events
                            if timestamp(event["available_at"]) < timestamp(cutoff)
                        ],
                        "result_path": str(result_path),
                        "input_sha256": frozen_hash,
                        "source_manifest_sha256": hashlib.sha256(
                            manifest_path.read_bytes()
                        ).hexdigest(),
                    },
                )
                counters["inputs"] += 1
                counters["generation_" + config["status"]] += 1
                counters["inputs_with_missing_code"] += bool(missing)
                counters["inputs_with_content_review_flags"] += bool(flags)
                counters["quarantined"] += exp_id in QUARANTINE
            inventory = next(item for item in trajectory_rows if item["trajectory_id"] == session)
            inventory.update(exp_ids=ids, status="checkpoint_prefixes_extracted")
            print(json.dumps({"session": session, "inputs_so_far": counters["inputs"]}), flush=True)
    finally:
        for handle in files.values():
            handle.close()
    with (out / "unscored_trajectories.jsonl").open("w") as handle:
        for inventory in trajectory_rows:
            if not inventory["exp_ids"]:
                value = unscored_trajectory(
                    mirror / "cells" / inventory["trajectory_id"], inventory
                )
                write_line(handle, value)
                counters["unscored_trajectory_prefixes"] += 1
    known = {item["id"] for item in checkpoints}
    extras = (
        []
        if sessions
        else sorted(
            path.stem
            for path in (mirror / "rescore10/results").glob("*.json")
            if path.stem not in known
        )
    )
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mirror": str(mirror),
        "counts": dict(counters),
        "trajectory_count": len(trajectory_rows),
        "trajectories_with_checkpoint_records": len(by_session),
        "recorder_trajectory_count": sum(item["recorder_available"] for item in trajectory_rows),
        "problems": problems,
        "result_ids_without_recorder_checkpoint": extras,
        "prefix_definition": "All retained code events strictly before first archive creation, grouped by chronological first-submission boundaries; not a causal training lineage. Later submissions reuse the existing archive and do not move the cutoff. Post-archive snapshot overwrites are excluded.",
        "label_definition": "Unweighted mean of ten run accuracies; equal test-set sizes make this equal total correct/(10*N). Not pass@10.",
        "generation_config_policy": "Archived metadata verified by matching fetch receipt URI and SHA256 if available; otherwise supplied-file observation, path-bound trajectory reconstruction or null. No inferred defaults.",
        "source_sha256": source_hashes,
        "source_code_unchanged_during_build": all(
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == value
            for name, value in source_hashes.items()
        ),
        "limitations": [
            "Extraction is not a semantic outcome-leakage review or proof of complete execution provenance.",
            "Comments/docstrings removed from parseable Python; executable literals are preserved and flagged heuristically.",
            "Obvious score-bearing echo/printf log strings are removed unless redirected; ambiguous numeric literals are retained and flagged.",
            "Recognized card-only serialization helpers are omitted from recipe inputs and retained as provenance hashes in the audit.",
            "Shell code and some source literals can encode prior measurements; review before modeling.",
            "No tool-result prose, card measurements, parent scores, or future code are supplied as input.",
            "Named code can be missing; imports, external datasets, environment state and unlogged mutations are not fully recovered.",
            "Checkpoint archive time is a retrospective cutoff, not the pre-training prediction time.",
            "Config reconstruction at archive time does not verify that rescore serving used identical effective settings.",
            "Missing labels/configs and quarantines are retained, not imputed or silently dropped.",
            "Split whole trajectories, not rows, to prevent shared-prefix train/test leakage.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out / "trajectories.json").write_text(json.dumps(trajectory_rows, indent=2) + "\n")
    (out / "README.md").write_text(render_readme(summary))
    return summary


def render_readme(summary):
    return (
        """# Code-prefix prediction dataset

One record per archived checkpoint; `trajectories.json` groups prefixes by source
trajectory. `inputs.jsonl` contains only input candidates and quality/provenance
keys, `labels.jsonl` contains targets, `examples.jsonl` joins them, and
`audit.jsonl` contains detailed evidence. Pass only a row's `input` to a predictor.
`unscored_trajectories.jsonl` preserves end-of-trajectory code/config candidates
for trajectories with no archived ten-run target; their outputs remain null.

The input is the chronological list `prefix_recipes[*].codes`, plus
`generation_config`, benchmark/base model, and fixed evaluation context. Code
entries contain literal file versions or shell commands, not eight-field recipe
summaries. Repeated versions and abandoned branches remain in the prefix.
Grouping by submission boundary is chronological, not proof of causal ancestry.

For run r: accuracy_r = correct_r / number_of_questions_r.
Target: avg_pass_rate = sum(accuracy_1, ..., accuracy_10) / 10.
With equal N questions per run: target = total_correct / (10 * N).
Rates are fractions in [0,1]; multiply by100 to display percentages.
This is repeated single-answer accuracy, NOT fraction solved at least once
(pass@10). The ten rates and dispersion are preserved with the mean.

This is an extraction dataset, not a certified model-ready corpus. Unknown raw
generation configs are null; inferred defaults are never passed off as files.
Review the per-row quality flags and the limitations below. Exact archived
configs may be supplied with `--metadata-dir <root>`, containing
`<exp_id>/generation_config.json` and the downloader's verification receipts.
Keep the same benchmark/tokenizer and explicitly resolve serving overrides and
stop behavior before using generation_config as the full effective policy.

## Counts

```json
"""
        + json.dumps(summary["counts"], indent=2)
        + "\n```\n\n## Limitations\n\n"
        + "\n".join("- " + item for item in summary["limitations"])
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR)
    parser.add_argument("--out", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--session", action="append", help="Optional explicit smoke-test session")
    args = parser.parse_args()
    print(json.dumps(build(args.mirror, args.out, args.metadata_dir, args.session), indent=2))


if __name__ == "__main__":
    main()
