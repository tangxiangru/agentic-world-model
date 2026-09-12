"""Bounded static recovery of additional observed code versions.

Never executes trajectory Python/shell and never reads final snapshots. Unsupported
transforms remain unresolved. Returned times are successful tool-result times.
Within-command ledger ordering is deliberately left to the caller. These are
bounded observed/static reconstructions, not a proof against arbitrary hidden
side effects in other commands or proof of the code ultimately executed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
from pathlib import Path

from tools.outcome_prediction.rpm_code_provenance import (
    explicit_shell_mutations,
    normalize_path,
)


def _code(path):
    return isinstance(path, str) and Path(path).suffix in {".py", ".sh", ".bash"}


def _literal_path(value, cwd):
    if not isinstance(value, str) or any(c in value for c in "$`*?{}"):
        raise ValueError("nonliteral path")
    return normalize_path(value, cwd)


def _lit(node):
    value = ast.literal_eval(node)
    if not isinstance(value, str):
        raise ValueError("expected literal string")  # noqa: TRY004 - invalid AST value
    return value


def _fragment(tokens):
    # Retain redirects for mutation auditing. Ambiguous quoted redirect tokens
    # may conservatively block recovery, but must not conceal a real write.
    return " ".join(
        t if re.fullmatch(r"\d*(?:>>?|<<?|>&)\d*", t) else shlex.quote(t) for t in tokens
    )


def _python_transform(source, known, cwd):
    """Only straight-line string reads/replaces/writes; reject other operations."""
    bindings, writes = {}, {}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in bindings:
            return bindings[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "read" and not node.args and not node.keywords:
                opened = node.func.value
                if (
                    isinstance(opened, ast.Call)
                    and isinstance(opened.func, ast.Name)
                    and opened.func.id == "open"
                ):
                    if len(opened.args) not in (1, 2) or opened.keywords:
                        raise ValueError("unsupported open")
                    if len(opened.args) == 2 and _lit(opened.args[1]) != "r":
                        raise ValueError("not a read")
                    path = _literal_path(_lit(opened.args[0]), cwd)
                    text = writes.get(path, known(path))
                    if text is None:
                        raise ValueError("source version unknown: " + path)
                    return text
            if node.func.attr == "replace" and len(node.args) in (2, 3) and not node.keywords:
                text = value(node.func.value)
                old, new = (_lit(a) for a in node.args[:2])
                if not old:
                    raise ValueError("empty replacement pattern")
                count = ast.literal_eval(node.args[2]) if len(node.args) == 3 else -1
                if not isinstance(count, int) or isinstance(count, bool):
                    raise ValueError("noninteger replacement count")
                return text.replace(old, new, count)
        raise ValueError("unsupported string expression")

    for stmt in ast.parse(source).body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports may have side effects")  # noqa: TRY004 - unsupported AST
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            if stmt.targets[0].id == "open":
                raise ValueError("shadowed open")
            bindings[stmt.targets[0].id] = value(stmt.value)
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            # Do not evaluate arbitrary arguments to print: those can mutate
            # files. Literal/variable-only diagnostic prints are harmless.
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == "print"
                and all(isinstance(a, (ast.Constant, ast.Name)) for a in call.args)
                and not call.keywords
            ):
                continue
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "write"
                and len(call.args) == 1
                and not call.keywords
            ):
                opened = call.func.value
                if (
                    isinstance(opened, ast.Call)
                    and isinstance(opened.func, ast.Name)
                    and opened.func.id == "open"
                ):
                    if len(opened.args) != 2 or opened.keywords or _lit(opened.args[1]) != "w":
                        raise ValueError("unsupported output open")
                    path = _literal_path(_lit(opened.args[0]), cwd)
                    if not _code(path):
                        raise ValueError("output is not code")
                    writes[path] = value(call.args[0])
                    continue
        raise ValueError("unsupported Python statement")
    return writes


def _sed_replace(text, expression):
    """Restricted POSIX BRE substitute: no groups, classes, backrefs or commands."""
    if len(expression) < 4 or expression[0] != "s" or expression[1].isalnum():
        raise ValueError("unsupported sed expression")
    delimiter = expression[1]
    fields, chunk, escaped = [], "", False
    for ch in expression[2:]:
        if ch == delimiter and not escaped:
            fields.append(chunk)
            chunk = ""
        else:
            chunk += ch
        if ch == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    fields.append(chunk)
    if len(fields) != 3 or fields[2] not in ("", "g"):
        raise ValueError("unsupported sed flags")
    pattern, replacement, flags = fields
    if not pattern:
        raise ValueError("empty sed pattern")
    regex, i = "", 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 1
            if i == len(pattern) or pattern[i] not in "[]\\.*^$/|":
                raise ValueError("unsupported BRE escape")
            regex += re.escape(pattern[i])
        elif ch in "[]":
            raise ValueError("BRE classes unsupported")
        elif ch in ".*^$":
            regex += ch
        else:
            regex += re.escape(ch)
        i += 1
    repl, i = "", 0
    while i < len(replacement):
        ch = replacement[i]
        if ch == "&":
            raise ValueError("sed match interpolation unsupported")
        if ch == "\\":
            i += 1
            if i == len(replacement) or replacement[i] not in "\\/&|":
                raise ValueError("sed replacement escape unsupported")
            ch = replacement[i]
        repl += ch
        i += 1
    compiled = re.compile(regex)
    return "".join(
        compiled.sub(lambda _: repl, line, count=0 if flags == "g" else 1)
        for line in text.splitlines(keepends=True)
    )


def recover_code_events(index, session_dir=None):
    """Return additional versions and unresolved evidence; session_dir is reserved.

    The caller must enforce its own cutoff using each version's time. All
    versions, including historical overwritten versions, are retained.
    """
    versions, unresolved = [], []
    raw_events = {}
    for number, line in enumerate(index.trace_path.open(), 1):
        match = re.match(r"^\[[^]]+\]\s+(\{.*)", line)
        if match:
            try:
                raw_events[number] = json.loads(match.group(1))
            except ValueError:
                pass

    def add(call, path, content, kind, at=None, extra=None):
        versions.append(
            {
                "path": path,
                "content": content,
                "time": at or call["result"]["time"],
                "call_line": call["line"],
                "result_line": call["result"]["line"],
                "tool_use_id": call["id"],
                "source_kind": kind,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                **(extra or {}),
            }
        )

    def known(path, call):
        candidates = [dict(v, path=p) for p, vs in index.snapshots.items() if p == path for v in vs]
        candidates.extend(v for v in versions if v["path"] == path)
        candidates = [
            v for v in candidates if v["time"] < call["time"] and v["result_line"] < call["line"]
        ]
        if not candidates:
            return None
        selected = max(candidates, key=lambda v: (v["time"], v["result_line"]))
        for intervening in index.calls:
            if not selected["result_line"] < intervening["line"] < call["line"]:
                continue
            if (
                intervening["name"] in {"Write", "Edit"}
                and normalize_path(intervening["input"].get("file_path"), index.cwd) == path
            ):
                return None
            if intervening["name"] == "Bash" and explicit_shell_mutations(
                intervening["input"].get("command", ""), path, index.cwd
            ):
                return None
        return selected["content"]

    for call in sorted(index.calls, key=lambda c: (c["line"], c["time"])):
        result = call.get("result")
        if not result or result.get("is_error"):
            continue
        event = raw_events.get(result["line"], {})
        structured = event.get("tool_use_result") or {}
        blocks = (event.get("message") or {}).get("content", [])
        matching = [b for b in blocks if isinstance(b, dict) and b.get("tool_use_id") == call["id"]]
        result_text = "\n".join(
            b.get("content", "") for b in matching if isinstance(b.get("content"), str)
        )
        path = normalize_path(call["input"].get("file_path"), index.cwd)
        if call["name"] == "Read" and _code(path):
            file = structured.get("file", {}) if isinstance(structured, dict) else {}
            content = file.get("content")
            if (
                file.get("filePath") == path
                and isinstance(content, str)
                and file.get("startLine") == 1
                and file.get("numLines") == file.get("totalLines")
                and file.get("numLines") in {len(content.splitlines()), len(content.split("\n"))}
            ):
                add(call, path, content, "complete_structured_read")
            else:
                unresolved.append(
                    {"call_line": call["line"], "path": path, "reason": "Read not proven complete"}
                )
        elif call["name"] == "Write" and _code(path) and not structured:
            success = "File created successfully at: " + path
            text = call["input"].get("content")
            if isinstance(text, str) and re.fullmatch(
                re.escape(success)
                + r"(?: \(file state is current in your context — no need to Read it back\))?",
                result_text.strip(),
            ):
                add(call, path, text, "text_confirmed_write")
        elif call["name"] == "Edit" and _code(path) and not structured:
            success = "The file " + path + " has been updated successfully."
            if not re.fullmatch(
                re.escape(success)
                + r"(?: \(file state is current in your context — no need to Read it back\))?",
                result_text.strip(),
            ):
                continue
            before = known(path, call)
            old, new = call["input"].get("old_string"), call["input"].get("new_string")
            replace_all = call["input"].get("replace_all", False)
            if (
                before is not None
                and isinstance(old, str)
                and old
                and isinstance(new, str)
                and (before.count(old) == 1 or replace_all and before.count(old) > 0)
            ):
                add(
                    call,
                    path,
                    before.replace(old, new, -1 if replace_all else 1),
                    "text_confirmed_edit",
                )
            else:
                unresolved.append(
                    {
                        "call_line": call["line"],
                        "path": path,
                        "reason": "Edit prior version or unique match unavailable",
                    }
                )
        elif call["name"] == "Bash":
            command = call["input"].get("command", "")
            if re.search(r"(?m)(?:^|[;&])\s*cd\s+", command):
                unresolved.append(
                    {
                        "call_line": call["line"],
                        "reason": "cwd changes or shell control flow unsupported",
                    }
                )
                continue
            if isinstance(structured, dict) and re.search(
                r"Traceback \(most recent call last\)|No such file or directory|Permission denied",
                str(structured.get("stderr", "")),
            ):
                unresolved.append(
                    {
                        "call_line": call["line"],
                        "reason": "shell reports possible transform failure",
                    }
                )
                continue
            staged = {}
            # Quoted Python heredocs only. Do not parse card/data heredocs as code.
            spans = []
            for match in re.finditer(
                r"(?m)^[^\n]*?\bpython\d?(?:\.\d+)?\s+-\s*<<\s*(['\"])(\w+)\1[^\n]*\n", command
            ):
                end = re.search(r"(?m)^" + re.escape(match[2]) + r"\s*$", command[match.end() :])
                if not end:
                    continue
                body = command[match.end() : match.end() + end.start()]
                spans.append((match.start(), match.end() + end.end()))
                header = command[match.start() : match.end()]
                if any(operator in header for operator in ("&&", "||", "|", "&")):
                    unresolved.append(
                        {
                            "call_line": call["line"],
                            "reason": "conditional/background Python transform unsupported",
                        }
                    )
                    continue
                if re.search(r"(?m)^\s*(?:cp\s|sed\s+-i\b)", command[: match.start()]):
                    unresolved.append(
                        {
                            "call_line": call["line"],
                            "reason": "mixed shell-before-Python dependency unsupported",
                        }
                    )
                    continue
                if not re.search(r"\.replace\(|\.read\(\)", body) or not re.search(
                    r"open\([^\n]+['\"]w['\"]", body
                ):
                    continue
                try:
                    staged.update(
                        _python_transform(
                            body,
                            lambda p, staged=staged, call=call: staged.get(p, known(p, call)),
                            index.cwd,
                        )
                    )
                except (SyntaxError, ValueError, TypeError, KeyError) as exc:
                    unresolved.append(
                        {
                            "call_line": call["line"],
                            "reason": "unsupported Python transform: " + str(exc),
                        }
                    )
            shell = command
            for start, end in sorted(spans, reverse=True):
                shell = shell[:start] + "\n" + shell[end:]
            if re.search(r"(?m)^\s*(?:if|for|while|until|case|function)\b", shell):
                unresolved.append(
                    {"call_line": call["line"], "reason": "shell control flow unsupported"}
                )
                continue
            if "<<" in shell:
                unresolved.append(
                    {
                        "call_line": call["line"],
                        "reason": "additional heredoc contents not inspected",
                    }
                )
                continue
            unrecognized = []
            for line in shell.splitlines():
                # Never process a line containing operators inside a quoted sed
                # expression by splitting raw text. shlex preserves its tokens.
                try:
                    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
                    lexer.whitespace_split = True
                    chunks, current, operators = [], [], []
                    for token in lexer:
                        if token in {";", "&&", "||", "|", "&"}:
                            operators.append(token)
                            chunks.append(current)
                            current = []
                        else:
                            current.append(token)
                    chunks.append(current)
                except ValueError:
                    continue
                if any(op in operators for op in ("&&", "||", "&", "|")) and any(
                    chunk and (chunk[0] == "cp" or chunk[:2] == ["sed", "-i"]) for chunk in chunks
                ):
                    unresolved.append(
                        {
                            "call_line": call["line"],
                            "reason": "conditional/background/piped shell transform unsupported",
                        }
                    )
                    unrecognized.extend(_fragment(chunk) for chunk in chunks)
                    continue
                for tokens in chunks:
                    recognized = False
                    try:
                        if len(tokens) == 3 and tokens[0] == "cp":
                            src, dst = (_literal_path(v, index.cwd) for v in tokens[1:])
                            content = staged.get(src, known(src, call))
                            if _code(dst) and content is not None:
                                staged[dst] = content
                                recognized = True
                        elif len(tokens) == 4 and tokens[:2] == ["sed", "-i"]:
                            dst = _literal_path(tokens[3], index.cwd)
                            content = staged.get(dst, known(dst, call))
                            if _code(dst) and content is not None:
                                staged[dst] = _sed_replace(content, tokens[2])
                                recognized = True
                    except (ValueError, re.error) as exc:
                        if "dst" in locals():
                            staged.pop(dst, None)
                        unresolved.append(
                            {
                                "call_line": call["line"],
                                "reason": "unsupported shell transform: " + str(exc),
                            }
                        )
                    if tokens and not recognized:
                        unrecognized.append(_fragment(tokens))
            for path in list(staged):
                if any(
                    explicit_shell_mutations(fragment, path, index.cwd) for fragment in unrecognized
                ):
                    staged.pop(path)
                    unresolved.append(
                        {
                            "call_line": call["line"],
                            "path": path,
                            "reason": "unrecognized same-call mutation prevents final-version recovery",
                        }
                    )
            for path, content in staged.items():
                add(call, path, content, "static_literal_transform")
    return {"versions": versions, "unresolved": unresolved}
