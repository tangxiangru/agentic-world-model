"""Path-bound, prefix-only generation_config.json evidence extraction.

Recorded code is parsed, never executed. A missing file is not a default config.
The result describes saved JSON, not the evaluator's effective sampling policy.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from tools.outcome_prediction.rpm_code_provenance import (
    TraceIndex,
    explicit_shell_mutations,
    normalize_path,
    timestamp,
)

CONFIG_NAME = "generation_config.json"
CONFIG_KEYS = {
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "max_new_tokens",
    "max_length",
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
    "stop_strings",
    "transformers_version",
    "cache_implementation",
    "_from_model_config",
}
LIMITATIONS = [
    "Trajectory reconstruction is not verification of the archived checkpoint bytes.",
    "Only successful path-bound reads, direct writes and statically resolvable patches are reconstructed; arbitrary scripts, dynamic paths and unlogged mutations are not executed or inferred.",
    "The JSON is not an effective-serving configuration: backend defaults, request overrides, tokenizer and stopping behavior are not resolved here.",
    "A config mutation after the cutoff is excluded even when its contents appear in a final snapshot.",
]
UNKNOWN = object()


def _jsonable_config(value):
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        return None
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return None


def _objects(text):
    """Read balanced JSON/Python dictionary literals; never evaluate source."""
    values = []
    for start, char in enumerate(text):
        if char != "{" or any(a <= start < b for a, b, _ in values):
            continue
        depth, quote, escaped = 0, None, False
        for end in range(start, len(text)):
            ch = text[end]
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    raw = text[start : end + 1]
                    try:
                        value = json.loads(raw)
                    except (ValueError, TypeError):
                        try:
                            value = ast.literal_eval(raw)
                        except (ValueError, SyntaxError):
                            break
                    value = _jsonable_config(value)
                    if value is not None and (not value or CONFIG_KEYS.intersection(value)):
                        values.append((start, end + 1, value))
                    break
    return [v for _, _, v in values]


def _config_path(path, cwd):
    if not isinstance(path, str) or any(x in path for x in ("*", "$", "?", "{", "}")):
        return None
    normalized = normalize_path(path, cwd)
    return normalized if normalized and posixpath.basename(normalized) == CONFIG_NAME else None


@dataclass
class ConfigValue:
    path: str
    content: dict | None
    partial: dict = field(default_factory=dict)
    removed: set = field(default_factory=set)
    dirty: bool = False


@dataclass
class FileValue:
    path: str
    mode: str = "r"


class _PythonEvidence:
    """Tiny static interpreter for literal JSON file I/O, not general Python."""

    def __init__(self, source, cwd, states):
        self.cwd, self.states, self.env = cwd, states, {}
        self.operations, self.print_paths, self.read_paths, self.write_paths = (
            [],
            set(),
            set(),
            set(),
        )
        self.current_line = 0
        self.unparsed_write = False
        try:
            tree = ast.parse(source)
        except SyntaxError:
            self.unparsed_write = CONFIG_NAME in source
            return
        self.run(tree.body)

    def value(self, node):
        if isinstance(node, ast.Name):
            return self.env.get(node.id, UNKNOWN)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Dict):
            keys, values = [self.value(k) for k in node.keys], [self.value(v) for v in node.values]
            if UNKNOWN in keys or any(v is UNKNOWN for v in values):
                return UNKNOWN
            try:
                return dict(zip(keys, values))
            except TypeError:
                return UNKNOWN
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self.value(x) for x in node.elts]
            return UNKNOWN if any(v is UNKNOWN for v in values) else values
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self.value(node.operand)
            return -val if isinstance(val, (int, float)) else UNKNOWN
        if isinstance(node, ast.BinOp):
            left, right = self.value(node.left), self.value(node.right)
            if isinstance(left, str) and isinstance(right, str):
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Div):
                    return posixpath.join(left, right)
            return UNKNOWN
        if isinstance(node, ast.JoinedStr):
            parts = [
                self.value(v.value) if isinstance(v, ast.FormattedValue) else self.value(v)
                for v in node.values
            ]
            return (
                "".join(str(v) for v in parts) if all(v is not UNKNOWN for v in parts) else UNKNOWN
            )
        if isinstance(node, ast.Subscript):
            obj, key = self.value(node.value), self.value(node.slice)
            if isinstance(obj, ConfigValue):
                obj = obj.content if obj.content is not None else obj.partial
            if isinstance(obj, dict) and key is not UNKNOWN:
                return obj.get(key, UNKNOWN)
            return UNKNOWN
        if not isinstance(node, ast.Call):
            return UNKNOWN
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        args = [self.value(a) for a in node.args]
        if name in {"Path", "str"} and args:
            return args[0] if isinstance(args[0], str) else UNKNOWN
        if name == "join" and args and all(isinstance(x, str) for x in args):
            return posixpath.join(*args)
        if name == "open" and args:
            path = _config_path(args[0], self.cwd)
            mode = (
                args[1]
                if len(args) > 1
                else next((self.value(k.value) for k in node.keywords if k.arg == "mode"), "r")
            )
            if path and isinstance(mode, str) and any(c in mode for c in "wax+"):
                self.write_paths.add(path)
            return FileValue(path, mode) if path and isinstance(mode, str) else UNKNOWN
        if name == "load" and args and isinstance(args[0], FileValue):
            path = args[0].path
            self.read_paths.add(path)
            previous = self.states.get(path)
            return ConfigValue(path, copy.deepcopy(previous.get("config")) if previous else None)
        if name == "loads" and args and isinstance(args[0], str):
            try:
                return json.loads(args[0])
            except ValueError:
                return UNKNOWN
        if name == "loads" and args and isinstance(args[0], ConfigValue):
            return args[0]
        if name == "dict":
            output = {}
            if args:
                value = args[0].content if isinstance(args[0], ConfigValue) else args[0]
                if not isinstance(value, dict):
                    return UNKNOWN
                output.update(value)
            for kw in node.keywords:
                value = self.value(kw.value)
                if kw.arg is None or value is UNKNOWN:
                    return UNKNOWN
                output[kw.arg] = value
            return output
        if name == "dumps" and args:
            if isinstance(args[0], ConfigValue):
                return args[0]
            value = args[0].content if isinstance(args[0], ConfigValue) else args[0]
            if _jsonable_config(value) is not None:
                return json.dumps(value)
        if name in {"read", "read_text"} and isinstance(node.func, ast.Attribute):
            receiver = self.value(node.func.value)
            path = (
                receiver.path
                if isinstance(receiver, FileValue)
                else _config_path(receiver, self.cwd)
            )
            if path:
                self.read_paths.add(path)
                previous = self.states.get(path)
                return ConfigValue(
                    path, copy.deepcopy(previous.get("config")) if previous else None
                )
        if name == "print":
            for value in args:
                if isinstance(value, ConfigValue) and not value.dirty:
                    self.print_paths.add(value.path)
                    self.operations.append(
                        {
                            "path": value.path,
                            "kind": "print_pending",
                            "statement_line": self.current_line,
                        }
                    )
                if isinstance(value, str) and (path := _config_path(value, self.cwd)):
                    self.print_paths.add(path)
                    self.operations.append(
                        {"path": path, "kind": "print_pending", "statement_line": self.current_line}
                    )
            return None
        if name == "dump" and len(args) >= 2 and isinstance(args[1], FileValue):
            value, destination = args[0], args[1]
            if not any(c in destination.mode for c in "wax+"):
                return UNKNOWN
            if isinstance(value, ConfigValue):
                config, partial, removed = value.content, value.partial, value.removed
            else:
                config, partial, removed = _jsonable_config(value), {}, set()
            self.operations.append(
                {
                    "path": destination.path,
                    "config": copy.deepcopy(config),
                    "partial_fields": copy.deepcopy(partial),
                    "removed_fields": sorted(removed),
                    "kind": "python_json_write",
                    "statement_line": self.current_line,
                }
            )
            if isinstance(value, ConfigValue):
                value.path = destination.path
                value.dirty = False
            return None
        if name in {"write", "write_text"} and args and isinstance(node.func, ast.Attribute):
            receiver = self.value(node.func.value)
            path = (
                receiver.path
                if isinstance(receiver, FileValue)
                else _config_path(receiver, self.cwd)
            )
            if path:
                try:
                    config = (
                        copy.deepcopy(args[0].content)
                        if isinstance(args[0], ConfigValue)
                        else _jsonable_config(json.loads(args[0]))
                        if isinstance(args[0], str)
                        else None
                    )
                except ValueError:
                    config = None
                self.write_paths.add(path)
                self.operations.append(
                    {
                        "path": path,
                        "config": config,
                        "kind": "python_text_write",
                        "statement_line": self.current_line,
                    }
                )
                return None
        if name in {"update", "pop"} and isinstance(node.func, ast.Attribute):
            receiver = self.value(node.func.value)
            if isinstance(receiver, ConfigValue):
                receiver.dirty = True
                dictionaries = [receiver.partial] + (
                    [receiver.content] if receiver.content is not None else []
                )
                if name == "pop" and args and isinstance(args[0], str):
                    for dictionary in dictionaries:
                        dictionary.pop(args[0], None)
                    receiver.removed.add(args[0])
                elif name == "update":
                    updates = args[0] if args and isinstance(args[0], dict) else {}
                    updates = {
                        **updates,
                        **{k.arg: self.value(k.value) for k in node.keywords if k.arg},
                    }
                    if any(v is UNKNOWN for v in updates.values()):
                        receiver.content = None
                    else:
                        for dictionary in dictionaries:
                            dictionary.update(updates)
                        receiver.removed.difference_update(updates)
                return None
            if isinstance(receiver, dict):
                if name == "update" and args and isinstance(args[0], dict):
                    receiver.update(args[0])
                elif name == "pop" and args:
                    receiver.pop(args[0], None)
                return None
        return UNKNOWN

    def run(self, body):
        for node in body:
            self.current_line = node.lineno
            if isinstance(node, ast.Assign):
                value = self.value(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.env[target.id] = value
                    elif isinstance(target, ast.Subscript):
                        receiver, key = self.value(target.value), self.value(target.slice)
                        if isinstance(receiver, ConfigValue) and isinstance(key, str):
                            receiver.dirty = True
                            if value is UNKNOWN:
                                receiver.content = None
                                receiver.partial.pop(key, None)
                            else:
                                receiver.partial[key] = value
                                receiver.removed.discard(key)
                                if receiver.content is not None:
                                    receiver.content[key] = value
                        elif isinstance(receiver, dict) and isinstance(key, str):
                            receiver[key] = value
            elif isinstance(node, ast.Expr):
                self.value(node.value)
            elif isinstance(node, ast.With):
                for item in node.items:
                    value = self.value(item.context_expr)
                    if isinstance(item.optional_vars, ast.Name):
                        self.env[item.optional_vars.id] = value
                self.run(node.body)
            elif isinstance(node, ast.For):
                values = self.value(node.iter)
                if (
                    isinstance(node.target, ast.Name)
                    and isinstance(values, list)
                    and len(values) <= 100
                ):
                    for value in values:
                        self.env[node.target.id] = value
                        self.run(node.body)
                elif CONFIG_NAME in ast.unparse(node):
                    self.unparsed_write = True
            elif isinstance(node, (ast.If, ast.Try, ast.While)):
                source = ast.unparse(node)
                if (
                    CONFIG_NAME in source
                    or self.read_paths
                    and re.search(r"\b(?:dump|write|update|pop|open)\b|\[[^]]+\]\s*=", source)
                ):
                    self.unparsed_write = True


def _python_source_spans(command):
    sources = []
    for match in re.finditer(
        r"(?:python[\d.]*)\s+(?:-\s*)?<<-?\s*(['\"]?)(\w+)\1[^\n]*\n(.*?)\n\2(?:\s|$)",
        command,
        re.DOTALL,
    ):
        sources.append((match.start(), match.end(), match.group(3)))
    # Preserve the offset of EACH invocation; str.find('python') incorrectly
    # assigns every -c command in a compound shell call to the first invocation.
    for match in re.finditer(
        r"(?<![\w/])(?:/[\w./-]+/)?python[\d.]*\s+-c\s+(\"(?:\\.|[^\"\\])*\"|'[^']*')",
        command,
        re.DOTALL,
    ):
        if any(start <= match.start() < end for start, end, _ in sources):
            continue
        try:
            source = shlex.split(match.group(1))[0]
        except (ValueError, IndexError):
            continue
        sources.append((match.start(), match.end(), source))
    return sorted(sources)


def _python_sources(command):
    return [(start, source) for start, _, source in _python_source_spans(command)]


def _ordered_direct_recovery(command, cwd, states, stdout):
    """Two bounded additions: foreground exact cp, and read-then-Python-patch.

    One unambiguous full stdout dictionary can seed a read-only snippet before a
    later snippet patches that same file. Shell commands are never executed.
    Background jobs and conditional shell execution are deliberately excluded.
    """
    if "&" in command or "||" in command:
        return None
    components = [
        (start, end, "python", source) for start, end, source in _python_source_spans(command)
    ]
    for match in re.finditer(
        r"cat\s*>\s*([^\s]+)\s*<<-?\s*(['\"]?)(\w+)\2[^\n]*\n(.*?)\n\3(?:\s|$)",
        command,
        re.DOTALL,
    ):
        path = _config_path(match.group(1).strip("\"'"), cwd)
        if path:
            try:
                config = _jsonable_config(json.loads(match.group(4)))
            except ValueError:
                config = None
            components.append((match.start(), match.end(), "literal_write", (path, config)))

    def inside(position):
        return any(start < position < end for start, end, _, _ in components)

    # Only cp SRC/generation_config.json DST/generation_config.json: no flags,
    # directories, expansions, globbing, redirections or conditional execution.
    copies = []
    for match in re.finditer(r"(?:^|[;\n])\s*cp\s+([^;\n|&]+)(?=$|[;\n])", command):
        if inside(match.start()):
            continue
        try:
            tokens = shlex.split(match.group(1), comments=True)
        except ValueError:
            continue
        if len(tokens) != 2:
            continue
        source, target = (_config_path(token, cwd) for token in tokens)
        if source and target:
            copies.append((match.start(), match.end(), "copy", (source, target)))
    components.extend(copies)
    for match in re.finditer(r"(?:^|[;\n])\s*cat\s+([^;\n|&]+)(?=$|[;\n])", command):
        if inside(match.start()):
            continue
        try:
            tokens = shlex.split(match.group(1), comments=True)
        except ValueError:
            continue
        if len(tokens) == 1 and (path := _config_path(tokens[0], cwd)):
            components.append((match.start(), match.end(), "read", path))
    components.sort(key=lambda item: item[0])
    if not copies and sum(kind == "python" for _, _, kind, _ in components) < 2:
        return None

    readers, writers = [], []
    for start, _, kind, value in components:
        if kind == "read":
            readers.append((start, value, True))
        elif kind == "python":
            parser = _PythonEvidence(value, cwd, states)
            for operation in parser.operations:
                if operation["kind"] == "print_pending":
                    readers.append(
                        (
                            start,
                            operation["path"],
                            not parser.write_paths and not parser.unparsed_write,
                        )
                    )
            if parser.write_paths:
                writers.append(start)
    objects = _objects(stdout)
    bound = readers[0] if len(readers) == 1 and len(objects) == 1 and readers[0][2] else None
    if not copies and not (bound and any(position > bound[0] for position in writers)):
        return None

    local_states, operations = dict(states), []

    def append(operation, position):
        operation = {**operation, "command_offset": position}
        operations.append(operation)
        local_states[operation["path"]] = operation

    for start, _, kind, value in components:
        if kind == "literal_write":
            path, config = value
            append({"path": path, "config": config, "kind": "literal_shell_json_write"}, start)
        elif kind == "copy":
            source, target = value
            previous = local_states.get(source)
            config = copy.deepcopy(previous.get("config")) if previous else None
            operation = {
                "path": target,
                "source_path": source,
                "config": config,
                "kind": "exact_foreground_config_copy",
            }
            if config is None:
                operation["reason"] = "successful exact copy from source with unknown full JSON"
            append(operation, start)
        elif kind == "python":
            parser = _PythonEvidence(value, cwd, local_states)
            handled = set()
            for operation in parser.operations:
                if operation["kind"] != "print_pending":
                    append(operation, start)
                    handled.add(operation["path"])
            unknown_paths = parser.write_paths - handled
            if parser.unparsed_write:
                unknown_paths.update(parser.read_paths | parser.write_paths)
            for path in unknown_paths:
                append(
                    {
                        "path": path,
                        "config": None,
                        "kind": "unresolved_python_mutation",
                        "reason": "unsupported Python mutation during ordered replay",
                    },
                    start,
                )
        if bound and start == bound[0]:
            append(
                {"path": bound[1], "config": objects[0], "kind": "path_bound_stdout_read"}, start
            )

    # A recognized write must not hide another sed/cp/open mutation elsewhere in
    # the shell call. Conservatively invalidate any such residual target write.
    residual = list(command)
    for start, end, _, _ in components:
        residual[start:end] = " " * (end - start)
    residual = "".join(residual)
    for path in {operation["path"] for operation in operations}:
        reasons = explicit_shell_mutations(residual, path, cwd)
        if reasons:
            append(
                {
                    "path": path,
                    "config": None,
                    "kind": "unresolved_mutation",
                    "reason": "; ".join(reasons),
                },
                len(command),
            )
    return operations


def _stdout_results(trace_path):
    results = {}
    for line in trace_path.open():
        match = re.match(r"^\[([^]]+)\]\s+(\{.*)", line)
        if not match:
            continue
        try:
            event = json.loads(match.group(2))
        except ValueError:
            continue
        message = event.get("message") or {}
        if not isinstance(message, dict):
            continue
        for block in message.get("content", []):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            structured = event.get("tool_use_result")
            if isinstance(structured, dict) and "stdout" in structured:
                output = structured.get("stdout") or ""
                interrupted = bool(structured.get("interrupted"))
            else:
                output, interrupted = block.get("content") or "", False
                if isinstance(output, list):
                    output = "\n".join(b.get("text", "") for b in output if isinstance(b, dict))
            file = structured.get("file") if isinstance(structured, dict) else None
            stderr = structured.get("stderr", "") if isinstance(structured, dict) else ""
            diagnostic = (
                (output if isinstance(output, str) else "")
                + "\n"
                + (stderr if isinstance(stderr, str) else "")
            )
            execution_error = bool(
                re.search(
                    r"Traceback \(most recent call last\)|Permission denied|No such file or directory|(?:SyntaxError|NameError|JSONDecodeError):",
                    diagnostic,
                )
            )
            results[block.get("tool_use_id")] = {
                "stdout": output if isinstance(output, str) else "",
                "interrupted": interrupted,
                "execution_error": execution_error,
                "file": file
                if isinstance(file, dict) and str(file.get("filePath", "")).endswith(CONFIG_NAME)
                else None,
            }
    return results


class GenerationIndex:
    def __init__(self, session_dir, trace_index=None):
        self.session_dir = Path(session_dir)
        self.trace_index = trace_index or TraceIndex(self.session_dir / "solve_out_sanitized.txt")
        self._declared_candidates = None
        self.events, states = [], {}
        results = _stdout_results(self.trace_index.trace_path)
        entries = []
        for path, snapshots in self.trace_index.snapshots.items():
            if posixpath.basename(path) != CONFIG_NAME:
                continue
            for snapshot in snapshots:
                try:
                    config = _jsonable_config(json.loads(snapshot["content"]))
                except ValueError:
                    config = None
                entries.append(
                    (
                        snapshot["time"],
                        snapshot["result_line"],
                        "direct",
                        {
                            "path": path,
                            "config": config,
                            "kind": "successful_" + snapshot["tool"].lower(),
                            "at": snapshot["at"],
                            "line": snapshot["call_line"],
                            "result_line": snapshot["result_line"],
                            "tool_use_id": snapshot["tool_use_id"],
                        },
                    )
                )
        for call in self.trace_index.calls:
            if call["name"] == "Read":
                path = _config_path(call["input"].get("file_path"), self.trace_index.cwd)
                result, file = call.get("result"), results.get(call["id"], {}).get("file")
                if (
                    path
                    and result
                    and not result["is_error"]
                    and file
                    and normalize_path(file.get("filePath"), self.trace_index.cwd) == path
                    and file.get("startLine") == 1
                    and file.get("numLines") == file.get("totalLines")
                ):
                    try:
                        config = _jsonable_config(json.loads(file["content"]))
                    except (KeyError, ValueError, TypeError):
                        config = None
                    if config is not None:
                        entries.append(
                            (
                                result["time"],
                                result["line"],
                                "direct",
                                {
                                    "path": path,
                                    "config": config,
                                    "kind": "complete_read_tool_result",
                                    "at": result["at"],
                                    "line": call["line"],
                                    "result_line": result["line"],
                                    "tool_use_id": call["id"],
                                },
                            )
                        )
            if call["name"] != "Bash":
                continue
            command = call["input"].get("command", "")
            if CONFIG_NAME not in command:
                continue
            result = call.get("result")
            at, line = (result["time"], result["line"]) if result else (call["time"], call["line"])
            entries.append((at, line, "bash", call))
        for at, line, kind, entry in sorted(entries, key=lambda x: (x[0], x[1])):
            if kind == "direct":
                event = {**entry, "time": at}
                self.events.append(event)
                states[event["path"]] = event
                continue
            call = entry
            command = call["input"].get("command", "")
            cwd = self.trace_index.cwd
            cd = re.match(r"\s*cd\s+([^;&\n]+)\s*&&", command)
            if cd:
                try:
                    names = shlex.split(cd.group(1))
                    if len(names) == 1 and "$" not in names[0]:
                        cwd = normalize_path(names[0], cwd)
                except ValueError:
                    pass
            output = results.get(call["id"], {})
            successful = (
                call.get("result")
                and not call["result"]["is_error"]
                and not output.get("interrupted")
                and not output.get("execution_error")
            )
            evidence = {
                "at": call["result"]["at"] if call.get("result") else call["at"],
                "line": call["line"],
                "result_line": line if call.get("result") else None,
                "tool_use_id": call["id"],
                "time": at,
            }
            operations, printed_paths, print_positions, write_positions = [], set(), [], []
            if successful:
                for position, source in _python_sources(command):
                    parser = _PythonEvidence(source, cwd, states)
                    for operation in parser.operations:
                        if operation["kind"] == "print_pending":
                            print_positions.append((position, operation["statement_line"]))
                        else:
                            operations.append(operation)
                            write_positions.append((position, operation["statement_line"]))
                    printed_paths.update(parser.print_paths)
                    unknown_paths = parser.write_paths - {o["path"] for o in operations}
                    if parser.unparsed_write:
                        unknown_paths.update(parser.read_paths | parser.write_paths)
                    for path in unknown_paths:
                        operations.append(
                            {
                                "path": path,
                                "config": None,
                                "kind": "unresolved_python_mutation",
                                "reason": "unsupported control flow or write in path-bound Python",
                            }
                        )
                        write_positions.append((position, 10**9))
                # A literal cat of one exact file can supply its complete contents.
                for match in re.finditer(r"(?:^|[;&|\n])\s*cat\s+([^;|&\n]+)", command):
                    try:
                        tokens = shlex.split(match.group(1))
                    except ValueError:
                        continue
                    paths = [
                        _config_path(t, cwd)
                        for t in tokens
                        if not t.startswith("-") and not t.startswith("2>")
                    ]
                    if len(paths) == 1 and paths[0]:
                        printed_paths.add(paths[0])
                        print_positions.append((match.start(), 0))
                objects = _objects(output.get("stdout", ""))
                if (
                    len(printed_paths) == 1
                    and len(objects) == 1
                    and len(print_positions) == 1
                    and (not write_positions or print_positions[0] >= max(write_positions))
                ):
                    operations.append(
                        {
                            "path": next(iter(printed_paths)),
                            "config": objects[0],
                            "kind": "path_bound_stdout_read",
                        }
                    )
                # Literal shell heredoc writes of JSON.
                for match in re.finditer(
                    r"cat\s*>\s*([^\s]+)\s*<<-?\s*(['\"]?)(\w+)\2[^\n]*\n(.*?)\n\3(?:\s|$)",
                    command,
                    re.DOTALL,
                ):
                    path = _config_path(match.group(1).strip("\"'"), cwd)
                    try:
                        config = _jsonable_config(json.loads(match.group(4)))
                    except ValueError:
                        config = None
                    if path:
                        operations.append(
                            {"path": path, "config": config, "kind": "literal_shell_json_write"}
                        )
                ordered = _ordered_direct_recovery(command, cwd, states, output.get("stdout", ""))
                if ordered is not None:
                    operations = ordered
            handled = {o["path"] for o in operations}
            mentioned_paths = set(states)
            for candidate in re.findall(r"[^\s'\"(){};,=<>]+generation_config\.json", command):
                if path := _config_path(candidate, cwd):
                    mentioned_paths.add(path)
            for path in mentioned_paths - handled:
                reasons = explicit_shell_mutations(command, path, cwd)
                if reasons:
                    operations.append(
                        {
                            "path": path,
                            "config": None,
                            "kind": "unresolved_mutation",
                            "reason": "; ".join(reasons),
                        }
                    )
            for operation in operations:
                event = {**operation, **evidence}
                self.events.append(event)
                states[event["path"]] = event


def index_session(session_dir, trace_index=None):
    return GenerationIndex(session_dir, trace_index)


def _literal_declarations(source):
    """Find explicit generation-file writes of literal dictionaries in code.

    This walks function bodies without claiming they executed. Ambiguous variable
    assignments are intentionally omitted. Results are audit candidates only.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    dictionaries, destinations = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            try:
                literal = _jsonable_config(ast.literal_eval(node.value))
            except (SyntaxError, ValueError, TypeError):
                literal = None
            for name in names:
                dictionaries.setdefault(name, []).append(literal)
                if CONFIG_NAME in ast.unparse(node.value):
                    destinations.add(name)
        elif isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name) and CONFIG_NAME in ast.unparse(
                    item.context_expr
                ):
                    destinations.add(item.optional_vars.id)

    def destination(node):
        return CONFIG_NAME in ast.unparse(node) or any(
            isinstance(child, ast.Name) and child.id in destinations for child in ast.walk(node)
        )

    declarations = []
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr != "dump"
            or len(node.args) < 2
            or not destination(node.args[1])
        ):
            continue
        try:
            config = _jsonable_config(ast.literal_eval(node.args[0]))
        except (SyntaxError, ValueError, TypeError):
            config = None
        if isinstance(node.args[0], ast.Name):
            values = dictionaries.get(node.args[0].id, [])
            if len(values) == 1:
                config = values[0]
        if config is not None and CONFIG_KEYS.intersection(config):
            declarations.append(
                {
                    "declaration_line": node.lineno,
                    "declared_generation_config": config,
                    "destination_expression": ast.unparse(node.args[1]),
                }
            )
    return declarations


def declared_config_candidates(index, cutoff):
    """Audit-only literals in successfully captured source versions later launched."""
    if index._declared_candidates is None:
        candidates = []
        scripts = {
            path: versions
            for path, versions in index.trace_index.snapshots.items()
            if path.endswith(".py") and any(CONFIG_NAME in s["content"] for s in versions)
        }
        for call in index.trace_index.calls:
            if call["name"] != "Bash":
                continue
            command = call["input"].get("command", "")
            # Match a runner invocation, not a grep/cat naming the same script.
            invocations = re.findall(
                r"(?:^|[;&\n])\s*(?:nohup\s+)?(?:python[\d.]*|torchrun|accelerate\s+launch)\s+([^;&\n]+)",
                command,
            )
            launched = set()
            for invocation in invocations:
                try:
                    tokens = shlex.split(invocation)
                except ValueError:
                    continue
                for token in tokens:
                    if token.endswith(".py"):
                        launched.add(normalize_path(token, index.trace_index.cwd))
                        break
            for path in launched & scripts.keys():
                record = index.trace_index.reconstruct(path, call["time"].isoformat())
                if record["status"] != "reconstructed":
                    continue
                for declaration in _literal_declarations(record["content"]):
                    candidates.append(
                        {
                            **declaration,
                            "script_path": path,
                            "launch_at": call["time"].isoformat(),
                            "launch_line": call["line"],
                            "launch_tool_use_id": call["id"],
                            "source_evidence": record["evidence"],
                            "source_sha256": record["sha256"],
                            "status": "declared_only_not_checkpoint_verified",
                        }
                    )
        index._declared_candidates = candidates
    before = timestamp(cutoff)
    return [
        copy.deepcopy(c) for c in index._declared_candidates if timestamp(c["launch_at"]) < before
    ]


def resolve_for_checkpoint(
    index, checkpoint_paths, cutoff, *, archived_config_path=None, archived_config_verified=False
):
    """Resolve exact-path evidence strictly before cutoff; never fill absent keys.

    ``checkpoint_paths`` are original source/archive directories (or JSON paths).
    A supplied local file must be independently associated with this checkpoint.
    Its timestamp/provenance must be validated by the caller before supplying it.
    """
    paths = []
    for path in checkpoint_paths:
        if not isinstance(path, str) or not path:
            continue
        path = path if path.endswith("/" + CONFIG_NAME) else posixpath.join(path, CONFIG_NAME)
        normalized = normalize_path(path, index.trace_index.cwd)
        if normalized and normalized not in paths:
            paths.append(normalized)
    result = {
        "generation_config": None,
        "status": "unknown",
        "checkpoint_config_paths": paths,
        "cutoff": cutoff,
        "evidence": [],
        "blockers": [],
        "partial_fields": {},
        "declared_config_candidates": declared_config_candidates(index, cutoff),
        "limitations": LIMITATIONS.copy(),
    }
    if archived_config_path is not None:
        path = Path(archived_config_path)
        try:
            config = _jsonable_config(json.loads(path.read_text()))
            if config is None:
                raise ValueError("not a JSON dictionary")
        except (OSError, ValueError) as error:
            result["blockers"].append({"reason": "local config unreadable", "detail": str(error)})
        else:
            result.update(
                generation_config=config,
                status="archived_file_verified"
                if archived_config_verified
                else "snapshot_file_observed",
            )
            result["evidence"] = [
                {
                    "kind": result["status"],
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ]
            return result
    before = timestamp(cutoff)
    candidates = [e for e in index.events if e["path"] in paths and e["time"] < before]
    if not candidates:
        result["blockers"].append(
            {"reason": "no complete path-bound config evidence strictly before cutoff"}
        )
        return result
    selected = max(
        enumerate(candidates),
        key=lambda pair: (pair[1]["time"], pair[1].get("result_line") or pair[1]["line"], pair[0]),
    )[1]
    result["evidence"] = [
        {k: v for k, v in event.items() if k not in {"config", "time", "partial_fields"}}
        for event in candidates
        if event["path"] == selected["path"]
    ]
    if selected.get("config") is not None:
        result.update(
            generation_config=copy.deepcopy(selected["config"]), status="trajectory_reconstructed"
        )
    else:
        result["partial_fields"] = copy.deepcopy(selected.get("partial_fields", {}))
        result["blockers"].append(
            {
                "reason": selected.get(
                    "reason", "latest path-bound mutation lacks full JSON evidence"
                ),
                "line": selected["line"],
            }
        )
    # A launched mutation may complete only after the cutoff (or have no result).
    # Its later successful capture must not allow us to reuse an earlier value.
    for call in index.trace_index.calls:
        if not (selected["time"] <= call["time"] < before):
            continue
        call_result = call.get("result")
        if call_result is not None and call_result["time"] < before:
            continue
        reasons = []
        if (
            call["name"] in {"Write", "Edit"}
            and normalize_path(call["input"].get("file_path"), index.trace_index.cwd)
            == selected["path"]
        ):
            reasons = ["config mutation not confirmed complete before cutoff"]
        elif call["name"] == "Bash":
            reasons = explicit_shell_mutations(
                call["input"].get("command", ""), selected["path"], index.trace_index.cwd
            )
        if reasons:
            result.update(generation_config=None, status="unknown")
            result["blockers"].append({"reason": "; ".join(reasons), "line": call["line"]})
    for failure in index.trace_index.result_failures:
        if failure["path"] == selected["path"] and selected["time"] <= failure["time"] < before:
            result.update(generation_config=None, status="unknown")
            result["blockers"].append({"reason": failure["reason"], "line": failure["line"]})
    return result
