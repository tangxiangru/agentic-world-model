"""Recover recorded script versions before a card's first submission.

Only successful timestamped Write/Edit results are content evidence. In particular,
Claude's Edit result includes the complete originalFile, so its resulting contents
can be reconstructed independently of older edits. Final wm snapshots are never
used as input or fallback. Shell commands are inspected, never executed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import posixpath
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

LIMITATIONS = [
    "Recovered text is evidence of a recorded file version, not a guarantee of the version executed by a later process.",
    "First submission is a decision-time proxy; actual process launch is not inferred by this helper.",
    "The shell audit detects explicit target writes but cannot rule out arbitrary indirect or self-modifying code, unlogged processes, or missing trace events.",
    "Only named training/data-builder scripts are covered; imports, configuration, data contents, and transitive dependencies are not reconstructed.",
]


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_path(value, cwd="/home/ben/task"):
    if not isinstance(value, str) or not value.strip():
        return None
    return posixpath.normpath(value if value.startswith("/") else posixpath.join(cwd, value))


def _python_targets(source, target, cwd):
    """Recognize literal/assigned-literal Python write destinations without eval."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    bindings = defaultdict(set)

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return bindings[node.id]
        if (
            isinstance(node, ast.Call)
            and node.args
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "Path"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "Path"
            )
        ):
            return literal(node.args[0])
        return set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name):
                    bindings[name.id].update(literal(node.value))
    reasons = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
            modes = (
                literal(node.args[1])
                if len(node.args) > 1
                else next((literal(kw.value) for kw in node.keywords if kw.arg == "mode"), {"r"})
            )
            if any(any(char in mode for char in "wax+") for mode in modes) and any(
                normalize_path(path, cwd) == target for path in literal(node.args[0])
            ):
                reasons.append("python open with write mode")
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"write_text", "write_bytes", "unlink", "rename", "replace"}
            and any(normalize_path(path, cwd) == target for path in literal(node.func.value))
        ):
            reasons.append("python Path mutation")
    return reasons


def explicit_shell_mutations(command, target, cwd="/home/ben/task"):
    """Conservative blockers for directly expressed writes to the target file."""
    reasons = []
    # Literal shell output redirection, including append. Reads using '<' do not count.
    for match in re.finditer(r"(?<![<])(?:>>?|&>)\s*(['\"]?)([^\s;'\"<>]+)\1", command):
        if normalize_path(match.group(2), cwd) == target:
            reasons.append("shell output redirection")
    for line in command.splitlines():
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            segments = [[]]
            for token in lexer:
                if token and all(char in ";&|" for char in token):
                    segments.append([])
                else:
                    segments[-1].append(token)
        except ValueError:
            continue
        for tokens in segments:
            if not tokens:
                continue
            for i, token in enumerate(tokens[:-1]):
                if token == "-c" and any("python" in x for x in tokens[:i]):
                    reasons.extend(_python_targets(tokens[i + 1], target, cwd))
            targets = [i for i, token in enumerate(tokens) if normalize_path(token, cwd) == target]
            if not targets:
                continue
            if any(token in {"sed", "perl"} for token in tokens) and any(
                re.match(r"-\w*i", token) for token in tokens
            ):
                reasons.append("in-place sed/perl command")
            if tokens[0] in {"cp", "mv", "rm", "install", "truncate", "tee"} and (
                tokens[0] in {"rm", "truncate", "tee"} or targets[-1] == len(tokens) - 1
            ):
                reasons.append("shell file mutation")
    # A shell heredoc's program is inspected statically. Embedded YAML/JSON strings
    # containing the target filename are not confused with actual write destinations.
    for match in re.finditer(
        r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[^\n]*\n(.*?)\n\2(?:\s|$)", command, re.DOTALL
    ):
        reasons.extend(_python_targets(match.group(3), target, cwd))
    return sorted(set(reasons))


class TraceIndex:
    def __init__(self, trace_path, cwd="/home/ben/task"):
        self.trace_path = Path(trace_path)
        self.cwd = cwd
        self.snapshots = defaultdict(list)
        self.calls = []
        self.pending = {}
        self.result_failures = []
        self.malformed_json_events = 0
        self.trace_sha256 = hashlib.sha256(self.trace_path.read_bytes()).hexdigest()
        for line_number, line in enumerate(self.trace_path.open(), 1):
            match = re.match(r"^\[([^]]+)\]\s+(\{.*)", line)
            if not match:
                continue
            try:
                event = json.loads(match.group(2))
            except json.JSONDecodeError:
                self.malformed_json_events += 1
                continue
            if event.get("type") == "system" and event.get("cwd"):
                self.cwd = event["cwd"]
            at = event.get("timestamp") or match.group(1)
            try:
                effective_time = max(timestamp(at), timestamp(match.group(1)))
            except (ValueError, TypeError):
                continue
            message = event.get("message") or {}
            if not isinstance(message, dict):
                continue
            for block in message.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    call = {
                        "id": block["id"],
                        "name": block.get("name"),
                        "input": block.get("input") or {},
                        "at": at,
                        "envelope_at": match.group(1),
                        "time": effective_time,
                        "line": line_number,
                        "result": None,
                    }
                    self.calls.append(call)
                    self.pending[call["id"]] = call
                elif block.get("type") == "tool_result":
                    call = self.pending.get(block.get("tool_use_id"))
                    if call is None:
                        continue
                    call["result"] = {
                        "at": at,
                        "time": effective_time,
                        "line": line_number,
                        "is_error": bool(block.get("is_error")),
                    }
                    if call["name"] not in {"Write", "Edit"} or block.get("is_error"):
                        continue
                    result = event.get("tool_use_result")
                    path = normalize_path(call["input"].get("file_path"), self.cwd)
                    try:
                        if (
                            not isinstance(result, dict)
                            or normalize_path(result.get("filePath"), self.cwd) != path
                        ):
                            raise ValueError("structured tool result missing or path mismatch")
                        if result.get("userModified"):
                            raise ValueError("tool reports concurrent user modification")
                        if call["name"] == "Write":
                            content = result.get("content")
                            if not isinstance(content, str) or content != call["input"].get(
                                "content"
                            ):
                                raise ValueError("Write request/result content mismatch")
                        else:
                            original, old, new = (
                                result.get(k) for k in ("originalFile", "oldString", "newString")
                            )
                            if not all(isinstance(x, str) for x in (original, old, new)) or not old:
                                raise ValueError("Edit lacks complete originalFile or edit strings")
                            count = original.count(old)
                            replace_all = bool(result.get("replaceAll"))
                            if replace_all != bool(call["input"].get("replace_all")):
                                raise ValueError("Edit request/result replacement scope disagrees")
                            if count == 0 or count > 1 and not replace_all:
                                raise ValueError("Edit replacement is absent or ambiguous")
                            if old != call["input"].get("old_string") or new != call["input"].get(
                                "new_string"
                            ):
                                raise ValueError("Edit request/result strings disagree")
                            content = original.replace(old, new, -1 if replace_all else 1)
                        self.snapshots[path].append(
                            {
                                "content": content,
                                "time": effective_time,
                                "at": at,
                                "envelope_at": match.group(1),
                                "result_line": line_number,
                                "call_line": call["line"],
                                "tool": call["name"],
                                "tool_use_id": call["id"],
                                "original_file_sha256": sha256(result["originalFile"])
                                if isinstance(result.get("originalFile"), str)
                                else None,
                            }
                        )
                    except ValueError as error:
                        self.result_failures.append(
                            {
                                "path": path,
                                "time": effective_time,
                                "line": line_number,
                                "reason": str(error),
                            }
                        )

    def reconstruct(self, script_path, cutoff):
        path = normalize_path(script_path, self.cwd)
        before = timestamp(cutoff)
        captures = [s for s in self.snapshots[path] if s["time"] < before]
        record = {
            "script_path": path,
            "cutoff": cutoff,
            "status": "unavailable",
            "content": None,
            "sha256": None,
            "evidence": None,
            "blockers": [],
            "limitations": LIMITATIONS,
            "trace_path": str(self.trace_path),
            "trace_sha256": self.trace_sha256,
        }
        if not captures:
            record["blockers"] = [
                {"reason": "no complete successful Write/Edit result strictly before cutoff"}
            ]
            return record
        selected = max(captures, key=lambda s: (s["time"], s["result_line"]))
        blockers = []
        for call in self.calls:
            if call["line"] <= selected["result_line"] or call["time"] >= before:
                continue
            if (
                call["name"] in {"Write", "Edit"}
                and normalize_path(call["input"].get("file_path"), self.cwd) == path
                and (call["result"] is None or call["result"]["time"] >= before)
            ):
                blockers.append(
                    {
                        "line": call["line"],
                        "at": call["at"],
                        "reason": "file mutation not confirmed complete before cutoff",
                    }
                )
            if call["name"] == "Bash":
                for reason in explicit_shell_mutations(
                    call["input"].get("command", ""), path, self.cwd
                ):
                    blockers.append({"line": call["line"], "at": call["at"], "reason": reason})
        blockers.extend(
            {k: v for k, v in failure.items() if k not in {"path", "time"}}
            for failure in self.result_failures
            if failure["path"] == path and selected["time"] <= failure["time"] < before
        )
        record.update(
            status="blocked" if blockers else "reconstructed",
            blockers=blockers,
            evidence={k: v for k, v in selected.items() if k not in {"content", "time"}},
            last_recorded_content_sha256=sha256(selected["content"]),
        )
        if not blockers:
            record.update(content=selected["content"], sha256=sha256(selected["content"]))
        return record


def reconstruct_script(trace_path, script_path, cutoff):
    return TraceIndex(trace_path).reconstruct(script_path, cutoff)


def reconstruct_card(index, card, cutoff):
    setup = card.get("setup") or {}
    requests = [("training", (setup.get("command") or {}).get("script"))]
    requests.extend(
        (f"data_builder_{i}", data.get("built_by"))
        for i, data in enumerate(setup.get("data") or [])
        if isinstance(data, dict)
    )
    result = []
    for role, path in requests:
        if isinstance(path, str) and path.strip():
            result.append({"role": role, **index.reconstruct(path, cutoff)})
        elif role == "training":
            result.append(
                {
                    "role": role,
                    "status": "unavailable",
                    "script_path": None,
                    "content": None,
                    "sha256": None,
                    "blockers": [{"reason": "card names no training script"}],
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/traj/raw/awm-gsm8k-trajectories")
    )
    parser.add_argument(
        "--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/analysis/rpm/code_provenance")
    )
    args = parser.parse_args()
    eligibility = {
        r["example_id"]: r["eligible"] and r["y"] is not None
        for r in (json.loads(s) for s in args.examples.read_text().splitlines() if s.strip())
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scripts").mkdir(exist_ok=True)
    cards, counts = [], Counter()
    for cell in sorted((args.raw_root / "cells").glob("r0-*")):
        index = TraceIndex(cell / "solve_out_sanitized.txt")
        for directory in sorted((cell / "wm/cards").glob("exp-*")):
            first = json.loads(min(directory.glob("record-*.json")).read_text())
            eid = f"{cell.name}/{directory.name}"
            scripts = reconstruct_card(index, first["card"], first["at"])
            for script in scripts:
                content = script.pop("content")
                if content is not None:
                    output = args.output_dir / "scripts" / f"{script['sha256']}.txt"
                    output.write_text(content)
                    script["reconstructed_content_file"] = str(output)
            eligible = eligibility.get(eid, False)
            training = next(s for s in scripts if s["role"] == "training")
            complete = bool(scripts) and all(s["status"] == "reconstructed" for s in scripts)
            counts["all_cards"] += 1
            counts["eligible_cards"] += eligible
            counts["all_training_reconstructed"] += training["status"] == "reconstructed"
            counts["eligible_training_reconstructed"] += (
                eligible and training["status"] == "reconstructed"
            )
            counts["all_named_scripts_reconstructed"] += complete
            counts["eligible_all_named_scripts_reconstructed"] += eligible and complete
            cards.append(
                {
                    "example_id": eid,
                    "eligible": eligible,
                    "cutoff": first["at"],
                    "all_named_scripts_reconstructed": complete,
                    "scripts": scripts,
                }
            )
    manifest = {
        "schema": "rpm-code-provenance-v1",
        "cutoff_policy": "successful tool result strictly before first card submission",
        "counts": dict(counts),
        "limitations": LIMITATIONS,
        "cards": cards,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"counts": dict(counts), "manifest": str(args.output_dir / "manifest.json")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
