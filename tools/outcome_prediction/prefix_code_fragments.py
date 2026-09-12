"""Recover complete prefix-observed source from overlapping head/tail reads.

Deliberately narrow: exact literal cat/head/tail commands, output delimiters,
same-prefix ls -la byte size, one overlap, and one byte-size-compatible newline
candidate. Never reads final snapshots or executes trajectory code.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shlex

from tools.outcome_prediction.rpm_code_provenance import normalize_path


def _tokens(command):
    if not isinstance(command, str) or "$(" in command or "`" in command:
        return []
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return []


def _path(value, cwd):
    if any(char in value for char in "$`*?{}"):
        return None
    path = normalize_path(value, cwd)
    return path if path and path.endswith((".py", ".sh", ".bash")) else None


def _count(tokens, kind):
    if len(tokens) == 2 and tokens[0] == kind and re.fullmatch(r"-[1-9][0-9]*", tokens[1]):
        return int(tokens[1][1:])
    if len(tokens) == 3 and tokens[:2] == [kind, "-n"] and re.fullmatch(r"[1-9][0-9]*", tokens[2]):
        return int(tokens[2])
    return None


def _head_spec(command, cwd):
    tokens = _tokens(command)
    # Optional timer command runs before both file-size and content observations.
    if len(tokens) >= 3 and tokens[0] == "bash" and tokens[2] == "&&":
        if _path(tokens[1], cwd) is None:
            return None
        tokens = tokens[3:]
    if len(tokens) < 14 or tokens[0] != "echo" or tokens[2:7] != ["&&", "ls", "-la", "&&", "echo"]:
        return None
    if tokens[8:10] != ["&&", "cat"] or tokens[11] != "|":
        return None
    path, count = _path(tokens[10], cwd), _count(tokens[12:], "head")
    if not path or not count or posixpath.dirname(path) != normalize_path(".", cwd):
        return None
    if tokens[1] == tokens[7] or "\n" in tokens[1] + tokens[7]:
        return None
    return {"path": path, "count": count, "ls_marker": tokens[1], "marker": tokens[7]}


def _readonly_suffix(tokens):
    if any(token in {"|", "||", "&"} or "<" in token or ">" in token for token in tokens):
        return False
    segments, current = [], []
    for token in tokens:
        if token in {"&&", ";"}:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    for segment in segments:
        if segment[0] == "do":
            segment = segment[1:]
        if not segment or segment[0] not in {"echo", "ls", "cat", "for", "done"}:
            return False
        if segment[0] == "for" and (len(segment) < 4 or segment[2] != "in"):
            return False
    return True


def _tail_spec(command, cwd):
    tokens = _tokens(command)
    if len(tokens) < 8 or tokens[0] != "cat" or tokens[2] != "|":
        return None
    try:
        delimiter = tokens.index("&&", 3)
    except ValueError:
        return None
    if len(tokens) < delimiter + 3:
        return None
    path, count = _path(tokens[1], cwd), _count(tokens[3:delimiter], "tail")
    if not path or not count or tokens[delimiter + 1] != "echo":
        return None
    marker = tokens[delimiter + 2]
    suffix = tokens[delimiter + 3 :]
    if suffix and (suffix[0] != "&&" or not _readonly_suffix(suffix[1:])):
        return None
    return {"path": path, "count": count, "marker": marker}


def _result_outputs(index):
    outputs = {}
    wanted = {
        call["result"]["line"]: call
        for call in index.calls
        if call.get("result") and call["name"] == "Bash"
    }
    for line_number, line in enumerate(index.trace_path.open(), 1):
        if line_number not in wanted:
            continue
        match = re.match(r"^\[[^]]+\]\s+(\{.*)", line)
        if not match:
            continue
        try:
            event = json.loads(match.group(1))
        except ValueError:
            continue
        result = event.get("tool_use_result")
        call = wanted[line_number]
        if (
            not isinstance(result, dict)
            or result.get("interrupted")
            or result.get("stderr")
            or call["result"]["is_error"]
        ):
            continue
        stdout = result.get("stdout")
        if isinstance(stdout, str) and "\r" not in stdout and "\x00" not in stdout:
            outputs[call["id"]] = stdout.splitlines()
    return outputs


def _head_fragment(spec, lines):
    if lines.count(spec["ls_marker"]) != 1 or lines.count(spec["marker"]) != 1:
        return None
    begin, end = lines.index(spec["ls_marker"]), lines.index(spec["marker"])
    if begin >= end:
        return None
    sizes = []
    for line in lines[begin + 1 : end]:
        fields = line.split(maxsplit=8)
        if (
            len(fields) == 9
            and re.fullmatch(r"-[rwxStTs-]{9}[.+@]?", fields[0])
            and fields[4].isdigit()
            and fields[8] == posixpath.basename(spec["path"])
        ):
            sizes.append(int(fields[4]))
    fragment = lines[end + 1 :]
    if len(sizes) != 1 or len(fragment) != spec["count"]:
        return None
    return fragment, sizes[0]


def _unchanged_between(index, head, tail, path):
    if head["result"]["time"] >= tail["time"] or head["result"]["line"] >= tail["line"]:
        return False
    for call in index.calls:
        if not head["line"] < call["line"] < tail["line"]:
            continue
        # Adjacent reads are the bounded use case. Any intervening shell call
        # could run a writer indirectly, so it is excluded, even if unrecognized.
        if call["name"] == "Bash":
            return False
        if (
            call["name"] in {"Write", "Edit"}
            and normalize_path(call["input"].get("file_path"), index.cwd) == path
        ):
            return False
    return True


def recover_fragment_versions(index):
    """Return versions compatible with prefix_code_recovery, or [] if unsure."""
    outputs, heads, versions = _result_outputs(index), [], []
    for call in index.calls:
        if call["name"] != "Bash" or call["id"] not in outputs:
            continue
        command, lines = call["input"].get("command", ""), outputs[call["id"]]
        spec = _head_spec(command, index.cwd)
        if spec:
            fragment = _head_fragment(spec, lines)
            if fragment:
                heads.append((call, spec, *fragment))
            continue
        spec = _tail_spec(command, index.cwd)
        if not spec or lines.count(spec["marker"]) != 1:
            continue
        tail = lines[: lines.index(spec["marker"])]
        if len(tail) != spec["count"]:
            continue
        for head_call, head_spec, head, size in heads:
            if head_spec["path"] != spec["path"] or not _unchanged_between(
                index, head_call, call, spec["path"]
            ):
                continue
            overlaps = [n for n in range(1, min(len(head), len(tail)) + 1) if head[-n:] == tail[:n]]
            if len(overlaps) != 1:
                continue
            overlap = overlaps[0]
            joined = "\n".join(head + tail[overlap:])
            candidates = [
                text for text in (joined, joined + "\n") if len(text.encode("utf-8")) == size
            ]
            if len(candidates) != 1:
                continue
            content = candidates[0]
            versions.append(
                {
                    "path": spec["path"],
                    "content": content,
                    "time": call["result"]["time"],
                    "call_line": call["line"],
                    "result_line": call["result"]["line"],
                    "tool_use_id": call["id"],
                    "source_kind": "overlapping_head_tail_size_verified",
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "observed_bytes": size,
                    "overlap_lines": overlap,
                    "fragment_evidence": [
                        {
                            "kind": "head_and_ls_size",
                            "call_line": head_call["line"],
                            "result_line": head_call["result"]["line"],
                            "tool_use_id": head_call["id"],
                            "available_at": head_call["result"]["time"].isoformat(),
                            "line_count": head_spec["count"],
                            "size_marker": head_spec["ls_marker"],
                            "content_marker": head_spec["marker"],
                        },
                        {
                            "kind": "tail",
                            "call_line": call["line"],
                            "result_line": call["result"]["line"],
                            "tool_use_id": call["id"],
                            "available_at": call["result"]["time"].isoformat(),
                            "line_count": spec["count"],
                            "end_marker": spec["marker"],
                        },
                    ],
                }
            )
    return versions
