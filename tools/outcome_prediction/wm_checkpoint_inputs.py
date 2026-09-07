"""Resolve declared checkpoint inputs without executing archived code.

Only first-plan argv and timestamp-reconstructed entrypoint code are considered.
Results establish static declarations, not executed bytes or scored identity.
The caller owns prospective eligibility and the separate official-grade join.
"""

from __future__ import annotations

import ast
import hashlib
import posixpath
import re
import shlex

from tools.outcome_prediction.wm_code_features import (
    UNKNOWN,
    _command_tokens,
    _Interpreter,
    _ReturnFlow,
    _script_index,
    _StopFlow,
    _Symbol,
)

WEIGHT_NAMES = {
    "model",
    "model_path",
    "model_name",
    "model_name_or_path",
    "init",
    "init_from",
    "init_model",
    "base",
    "base_model",
    "checkpoint",
    "ckpt",
    "src",
    "weights",
    "adapter",
    "adapter_path",
    "peft_model",
}
MERGE_NAMES = {"models", "srcs", "src", "inputs", "a", "b", "sources", "ckpt", "checkpoints"}
RESUME_NAMES = {"resume", "resume_from_checkpoint", "resume_checkpoint"}


def _name(value):
    return value.lstrip("-").replace("-", "_")


def _weighted_merge_arguments(tree):
    """Recognize ``for s in args.ckpt: s.rsplit(':', 1)`` source syntax."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.iter, ast.Attribute):
            continue
        for statement in node.body:
            for call in ast.walk(statement):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == node.target.id
                    and call.func.attr in {"split", "rsplit", "partition", "rpartition"}
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == ":"
                ):
                    names.add(node.iter.attr)
    return names


def _literal_path(value, cwd, base):
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if re.search(r"[$`{}*?<>\n\r]|\s\+\s", value) or value.startswith("~"):
        return None
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return None
    if base and (
        value == base
        or re.search(
            r"(?:^|/)models--" + re.escape(base.replace("/", "--")) + r"/snapshots/[0-9a-f]{40}/?$",
            value,
        )
    ):
        return {"path": value.rstrip("/"), "base_model": base}
    if value.startswith("/"):
        return {"path": posixpath.normpath(value)}
    if not isinstance(cwd, str) or not cwd.startswith("/") or re.search(r"[$`{}*?]", cwd):
        return None
    return {"path": posixpath.normpath(posixpath.join(cwd, value))}


def _unwrap(command):
    """Accept literal env/cd wrappers, never expansion or arbitrary shell logic."""
    argv = _command_tokens(command)
    cwd = command.get("cwd")
    wrappers = []
    for _ in range(4):
        if not argv:
            return [], cwd, wrappers, "missing_command"
        if posixpath.basename(argv[0]) in {"bash", "sh", "zsh"}:
            if len(argv) != 3 or argv[1] not in {"-c", "-lc"}:
                return [], cwd, wrappers, "unsupported_shell_wrapper"
            try:
                argv = shlex.split(argv[2])
            except ValueError:
                return [], cwd, wrappers, "malformed_shell_wrapper"
            wrappers.append("literal_shell_command")
            continue
        if argv[0] == "cd":
            if len(argv) < 4 or argv[2] != "&&":
                return [], cwd, wrappers, "unsupported_cd_wrapper"
            path = _literal_path(argv[1], cwd, None)
            if path is None:
                return [], cwd, wrappers, "dynamic_working_directory"
            cwd, argv = path["path"], argv[3:]
            wrappers.append("literal_cd")
            continue
        if posixpath.basename(argv[0]) == "env":
            argv = argv[1:]
            while argv and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=[^\n\r]*", argv[0]):
                argv = argv[1:]
            wrappers.append("env_assignments")
            continue
        break
    if any(re.search(r"[;&|<>`\n\r]|\$", token) for token in argv):
        return [], cwd, wrappers, "dynamic_or_compound_command"
    return argv, cwd, wrappers, None


def _parse_arguments(specifications, defaults, argv):
    """Bounded argparse semantics including nargs; no parser or script is run."""
    result, lookup, positionals, sources = {}, {}, [], {}
    for flags, spec in specifications:
        dest = spec.get("dest") or next(
            (f[2:] for f in flags if f.startswith("--")), flags[0].lstrip("-")
        )
        if not isinstance(dest, str):
            return {}, {}, "dynamic_argument_destination"
        dest = _name(dest)
        action = spec.get("action", "store")
        value = defaults.get(
            dest,
            spec.get(
                "default",
                action == "store_false" if action in {"store_true", "store_false"} else None,
            ),
        )
        result.setdefault(dest, value)
        sources.setdefault(dest, "argparse_default")
        item = (dest, spec)
        for flag in flags:
            if flag.startswith("-"):
                if flag in lookup and lookup[flag] != item:
                    return {}, {}, "conflicting_argument_declarations"
                lookup[flag] = item
        if not any(f.startswith("-") for f in flags):
            positionals.append(item)

    def convert(value, spec):
        kind = spec.get("type")
        if kind is None:
            return value
        if not isinstance(kind, _Symbol) or kind.name not in {"str", "int", "float"}:
            return UNKNOWN
        try:
            return {"str": str, "int": int, "float": float}[kind.name](value)
        except (TypeError, ValueError):
            return UNKNOWN

    def consume(index, spec, inline=None):
        nargs = spec.get("nargs")
        action = spec.get("action", "store")
        if action in {"store_true", "store_false", "store_const"}:
            if inline is not None:
                return UNKNOWN, index, "value_for_flag_action"
            return (
                (action == "store_true" if action != "store_const" else spec.get("const")),
                index,
                None,
            )
        if action not in {"store", "append"}:
            return UNKNOWN, index, "unsupported_argument_action"
        if nargs is None:
            minimum = maximum = 1
        elif isinstance(nargs, int) and not isinstance(nargs, bool) and 0 <= nargs <= 100:
            minimum = maximum = nargs
        elif nargs in {"?", "*", "+"}:
            minimum, maximum = (1 if nargs == "+" else 0), (1 if nargs == "?" else len(argv))
        else:
            return UNKNOWN, index, "unsupported_nargs"
        values = [] if inline is None else [inline]
        while len(values) < maximum and index < len(argv):
            token = argv[index]
            if token == "--" or token.partition("=")[0] in lookup or token.startswith("--"):
                break
            values.append(token)
            index += 1
        if len(values) < minimum or len(values) > maximum:
            return UNKNOWN, index, "missing_argument_value"
        values = [convert(v, spec) for v in values]
        if nargs is None or nargs == "?":
            return (values[0] if values else spec.get("const")), index, None
        return values, index, None

    index, positional_index = 0, 0
    supplied_specs = set()
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return {}, {}, "argument_terminator_not_supported"
        flag, sep, inline = token.partition("=")
        if flag in lookup:
            dest, spec = lookup[flag]
            supplied_specs.add(id(spec))
            index += 1
            value, index, reason = consume(index, spec, inline if sep else None)
        elif token.startswith("-"):
            return {}, {}, "undeclared_command_option"
        elif positional_index < len(positionals):
            dest, spec = positionals[positional_index]
            positional_index += 1
            value, index, reason = consume(index, spec)
        else:
            return {}, {}, "undeclared_positional_argument"
        if reason:
            return {}, {}, reason
        if spec.get("action") == "append":
            old = result.get(dest)
            value = ([] if old is None else list(old) if isinstance(old, list) else [old]) + [value]
        result[dest], sources[dest] = value, "parsed_argv"
    for dest, spec in positionals[positional_index:]:
        if spec.get("nargs") not in {"?", "*"}:
            return {}, {}, "missing_positional_argument"
        result[dest] = spec.get("default")
    for flags, spec in specifications:
        if (
            spec.get("required") is True
            and any(f.startswith("-") for f in flags)
            and id(spec) not in supplied_specs
        ):
            return {}, {}, "missing_required_argument"
        dest = _name(
            spec.get("dest")
            or next((f[2:] for f in flags if f.startswith("--")), flags[0].lstrip("-"))
        )
        if sources.get(dest) == "argparse_default" and isinstance(result.get(dest), str):
            result[dest] = convert(result[dest], spec)
    return result, sources, None


class _InputInterpreter(_Interpreter):
    def __init__(self, argv, cwd, base_model):
        super().__init__(argv)
        self.cwd, self.base_model = cwd, base_model
        self.path_operations = []
        self.arguments = []
        self.loads, self.resumes, self.input_errors = [], [], []
        self.runtime_guards = []
        self.trainer_classes = {"Trainer", "SFTTrainer", "GRPOTrainer", "DPOTrainer"}
        self.custom_trainers = set()

    def bound_path(self, value):
        parsed = _literal_path(value, self.cwd, self.base_model)
        return parsed["path"] if parsed else UNKNOWN

    def checked_keywords(self, node, env):
        values = {}
        for keyword in node.keywords:
            value = self.value(keyword.value, env)
            supplied = {keyword.arg: value} if keyword.arg is not None else value
            if not isinstance(supplied, dict) or any(not isinstance(k, str) for k in supplied):
                return None
            if set(values) & set(supplied):
                return None
            values.update(supplied)
        return values

    def block(self, nodes, env):
        # An unevaluated assertion may stop the program but cannot choose a
        # different input path. Resolve intended inputs conditionally on it
        # passing, and retain that assumption in evidence. Branches still use
        # the conservative inherited interpreter.
        for index, node in enumerate(nodes):
            if isinstance(node, ast.ClassDef):
                bases = [self.value(base, env) for base in node.bases]
                overrides = {
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                supported = bool(bases) and all(
                    isinstance(base, _Symbol)
                    and (
                        base.name in self.custom_trainers
                        or base.name.startswith(("transformers.", "trl."))
                        and base.name.rsplit(".", 1)[-1] in self.trainer_classes
                    )
                    for base in bases
                )
                if (
                    supported
                    and not node.decorator_list
                    and not overrides.intersection(
                        {"train", "__init__", "__getattr__", "__getattribute__"}
                    )
                ):
                    symbol = "checkpoint.CustomTrainer." + node.name
                    self.custom_trainers.add(symbol)
                    env[node.name] = _Symbol(symbol)
                else:
                    env[node.name] = UNKNOWN
                continue
            if isinstance(node, ast.Assert):
                value = self.value(node.test, env)
                if isinstance(value, (str, bool, int, float)) and not value:
                    self.input_errors.append("statically_failed_assertion")
                    raise _StopFlow()
                if value is UNKNOWN:
                    self.runtime_guards.append(node.lineno)
                continue
            try:
                super().block([node], env)
            except _StopFlow:
                self.uncertain(nodes[index + 1 :], env)
                raise
        return UNKNOWN

    def assign(self, target, value, env):
        if isinstance(target, ast.Attribute) and target.attr in {
            "from_pretrained",
            "train",
            "load_adapter",
            "load_state_dict",
        }:
            self.input_errors.append("dynamic_weight_method_override")
        super().assign(target, value, env)

    def parse_arguments(self, parser):
        values, sources, reason = _parse_arguments(parser.arguments, parser.defaults, self.argv)
        if parser.unresolved:
            reason = "conditional_argument_declarations"
        if reason:
            self.input_errors.append(reason)
            return dict.fromkeys(values, UNKNOWN)
        self.arguments.append((values, sources))
        return values

    def call(self, node, env):
        func = self.value(node.func, env)
        full = func.name if isinstance(func, _Symbol) else ""
        base = full.rsplit(".", 1)[-1] or (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else ""
        )
        owner = full.rpartition(".")[0]
        if full == "os.chdir":
            values = [self.value(a, env) for a in node.args]
            parsed = (
                _literal_path(values[0], self.cwd, None)
                if len(values) == 1 and not node.keywords
                else None
            )
            self.cwd = parsed["path"] if parsed else None
            self.path_operations.append(
                {"source": "declared_chdir", "line": node.lineno, "cwd": self.cwd}
            )
            return UNKNOWN
        if (
            base == "from_pretrained"
            and re.search(r"Model|ForCausalLM|ForConditionalGeneration", owner)
            and not re.search(r"Tokenizer|Processor|Config", owner)
        ):
            kwargs = self.keyword_values(node, env)
            args = [self.value(arg, env) for arg in node.args]
            value = (
                args[1]
                if "PeftModel" in owner and len(args) > 1
                else args[0]
                if args
                else kwargs.get("pretrained_model_name_or_path", kwargs.get("model_name", UNKNOWN))
            )
            if any(k.arg is None and self.value(k.value, env) is UNKNOWN for k in node.keywords):
                value = UNKNOWN
            self.loads.append((self.bound_path(value), node.lineno, "model_from_pretrained"))
            return _Symbol("checkpoint.Model")
        if (
            base in self.trainer_classes and full.startswith(("transformers.", "trl."))
        ) or full in self.custom_trainers:
            kwargs = self.keyword_values(node, env)
            if isinstance(kwargs.get("model"), str):
                self.loads.append(
                    (self.bound_path(kwargs["model"]), node.lineno, "trainer_model_argument")
                )
            return _Symbol("checkpoint.Trainer")
        if full == "checkpoint.Trainer.train":
            values = self.checked_keywords(node, env)
            value = (
                UNKNOWN
                if values is None or (node.args and "resume_from_checkpoint" in values)
                else self.value(node.args[0], env)
                if node.args
                else values.get("resume_from_checkpoint")
            )
            if value is not None and value is not False:
                self.resumes.append(
                    (self.bound_path(value), node.lineno, "trainer_resume_from_checkpoint")
                )
            return UNKNOWN
        if base == "train" and any(k.arg == "resume_from_checkpoint" for k in node.keywords):
            self.resumes.append((UNKNOWN, node.lineno, "unverified_trainer_resume_receiver"))
            return UNKNOWN
        if full == "checkpoint.Model.load_adapter":
            kwargs = self.keyword_values(node, env)
            value = (
                self.value(node.args[0], env) if node.args else kwargs.get("peft_model_id", UNKNOWN)
            )
            self.loads.append((self.bound_path(value), node.lineno, "model_load_adapter"))
            return UNKNOWN
        if full == "checkpoint.Model.load_state_dict":
            self.loads.append((UNKNOWN, node.lineno, "state_dict_checkpoint_source_unresolved"))
            return UNKNOWN
        if full == "peft.get_peft_model" and node.args:
            first = self.value(node.args[0], env)
            if isinstance(first, _Symbol) and first.name == "checkpoint.Model":
                return first
        if full == "checkpoint.Model.merge_and_unload":
            return _Symbol("checkpoint.Model")
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            values = self.checked_keywords(node, env)
            names = [arg.arg for arg in func.args.args]
            supplied = set(names[: len(node.args)])
            required = set(names[: len(names) - len(func.args.defaults)])
            if (
                isinstance(func, ast.AsyncFunctionDef)
                or func.decorator_list
                or func.args.posonlyargs
                or func.args.vararg
                or func.args.kwarg
                or func.args.kwonlyargs
                or self.depth >= 4
                or values is None
                or len(node.args) > len(names)
                or any(isinstance(arg, ast.Starred) for arg in node.args)
                or bool(set(values or {}) - set(names))
                or bool(supplied & set(values or {}))
                or bool(required - supplied - set(values or {}))
            ):
                self.input_errors.append("unsupported_or_ambiguous_local_helper_call")
                return UNKNOWN
        return super().call(node, env)

    def uncertain(self, nodes, env):
        for statement in nodes:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                func = self.value(node.func, env)
                full = func.name if isinstance(func, _Symbol) else ""
                if full == "os.chdir":
                    self.cwd = None
                if (
                    full.endswith(".from_pretrained")
                    and re.search(r"Model|ForCausalLM|ForConditionalGeneration", full)
                    and not re.search(r"Tokenizer|Processor|Config", full)
                ):
                    self.loads.append((UNKNOWN, node.lineno, "conditional_model_load"))
                if full == "checkpoint.Trainer.train" and any(
                    k.arg == "resume_from_checkpoint" for k in node.keywords
                ):
                    self.resumes.append((UNKNOWN, node.lineno, "conditional_resume"))
                if full in {"checkpoint.Model.load_adapter", "checkpoint.Model.load_state_dict"}:
                    self.loads.append((UNKNOWN, node.lineno, "conditional_checkpoint_weight_load"))
        super().uncertain(nodes, env)


def resolve_inputs(row):
    """Return statically supported checkpoint paths and auditable uncertainty.

    ``row`` is an inventory-shaped dict containing ``model_input``, or that
    model-input dict itself. Labels, observations, origins and final snapshots
    are not accessed. For unresolved cases, ``inputs`` may contain candidates,
    but callers MUST NOT treat them as confirmed inputs.
    """
    source = row.get("model_input", row)
    setup = (source.get("plan") or {}).get("setup") or {}
    command = setup.get("command") or {}
    base = setup.get("base_model") or (source.get("task") or {}).get("base_model")
    if not isinstance(base, str):
        base = None
    evidence, reasons, candidates = [], [], []

    def finish(values=(), reason=None):
        if reason:
            reasons.append(reason)
        for value, line, kind in values:
            values_here = value if isinstance(value, (list, tuple)) else [value]
            for raw in values_here:
                parsed = _literal_path(raw, cwd, base)
                if parsed is None:
                    reasons.append("unknown_or_nonliteral_checkpoint")
                else:
                    candidates.append(parsed)
                    evidence.append(
                        {"source": kind, "line": line, "raw": raw, "path": parsed["path"]}
                    )
        unique = {
            ("base", p["base_model"]) if "base_model" in p else ("path", p["path"]): p
            for p in candidates
        }
        inputs = list(unique.values())
        declared_parent = _literal_path(
            (setup.get("parent_checkpoint") or {}).get("path"), command.get("cwd"), base
        )
        if declared_parent and inputs:
            parent_identity = (
                ("base", declared_parent["base_model"])
                if "base_model" in declared_parent
                else ("path", declared_parent["path"])
            )
            if parent_identity not in unique:
                evidence.append(
                    {
                        "source": "structured_parent_disagrees_with_stronger_input",
                        "declared_path": declared_parent["path"],
                        "resolved_paths": [p["path"] for p in inputs],
                    }
                )
        status = (
            "unresolved"
            if reasons or not inputs
            else "multiple"
            if len(inputs) > 1
            else "base"
            if "base_model" in inputs[0]
            else "single"
        )
        return {
            "status": status,
            "inputs": inputs,
            "evidence": evidence,
            "reasons": sorted(set(reasons)),
        }

    argv, cwd, wrappers, error = _unwrap(command)
    evidence.extend({"source": wrapper} for wrapper in wrappers)
    if error:
        return finish(reason=error)
    index = _script_index(argv)
    if index is None:
        return finish(reason="unsupported_or_ambiguous_launcher")
    script = _literal_path(argv[index], cwd, None)
    declared_script = (
        _literal_path(command.get("script"), command.get("cwd"), None)
        if command.get("script")
        else None
    )
    if script is None:
        return finish(reason="invalid_entrypoint_path")
    if declared_script and script != declared_script:
        return finish(reason="command_script_conflict")
    merge = bool(
        re.search(
            r"merge|soup|averag",
            str((setup.get("method") or {}).get("family", ""))
            + " "
            + posixpath.basename(script["path"]),
            re.IGNORECASE,
        )
    )
    code = [
        c
        for c in source.get("code") or []
        if c.get("status") == "reconstructed"
        and isinstance(c.get("content"), str)
        and _literal_path(c.get("script_path"), cwd, None) == script
    ]
    contents = {c["content"] for c in code}
    if len(contents) > 1:
        return finish(reason="conflicting_reconstructed_entrypoint")
    interpreter = None
    weighted_merge_arguments = set()
    if contents:
        content = next(iter(contents))
        evidence.append(
            {
                "source": "reconstructed_entrypoint",
                "path": script["path"],
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
        try:
            tree = ast.parse(content)
            weighted_merge_arguments = _weighted_merge_arguments(tree)
            if sum(1 for _ in ast.walk(tree)) > 100_000:
                return finish(reason="entrypoint_ast_too_large")
            interpreter = _InputInterpreter(argv[index + 1 :], cwd, base)
            try:
                interpreter.block(tree.body, {"__name__": "__main__"})
            except (_ReturnFlow, _StopFlow):
                pass
        except (SyntaxError, ValueError, RecursionError, TypeError):
            return finish(reason="entrypoint_static_parse_failed")
        if interpreter.input_errors:
            reasons.extend(interpreter.input_errors)
            return finish()
        evidence.extend(
            {"source": "input_conditional_on_runtime_assertion_passing", "line": line}
            for line in interpreter.runtime_guards
        )
        evidence.extend(interpreter.path_operations)
        if interpreter.resumes:
            evidence.append({"source": "verified_trainer_resume_overrides_initial_model"})
            return finish(interpreter.resumes)
        if interpreter.loads:
            return finish(interpreter.loads)
        if interpreter.path_operations or interpreter.cwd != cwd:
            return finish(reason="checkpoint_input_not_verified_after_chdir")

    declared = []
    if interpreter and interpreter.arguments:
        for values, sources in interpreter.arguments:
            names = MERGE_NAMES if merge else WEIGHT_NAMES
            for name, value in values.items():
                if name in RESUME_NAMES and value not in (None, False):
                    return finish(reason="resume_precedence_not_verified_in_code")
                if name in names and value is not None:
                    if merge and name in weighted_merge_arguments:
                        items = value if isinstance(value, list) else [value]
                        decoded = []
                        for item in items:
                            match = (
                                re.fullmatch(r"(.+):[+-]?(?:\d+(?:\.\d*)?|\.\d+)", item)
                                if isinstance(item, str)
                                else None
                            )
                            if match is None:
                                return finish(reason="invalid_weighted_merge_checkpoint")
                            decoded.append(match[1])
                        value = decoded
                        evidence.append(
                            {
                                "source": "code_verified_checkpoint_weight_separator",
                                "argument": name,
                            }
                        )
                    declared.append((value, None, sources[name] + ":" + name))
        if declared:
            # Comma-separated source lists are explicit merge syntax only.
            if merge:
                declared = [
                    (v.split(",") if isinstance(v, str) else v, line, kind)
                    for v, line, kind in declared
                ]
            return finish(declared)

    if interpreter and interpreter.loads:
        return finish(interpreter.loads)

    # Without supported parser flow, use only exact known option spellings.
    # Single options consume ONE token, never arbitrary later positionals.
    tail, cursor = argv[index + 1 :], 0
    while cursor < len(tail):
        token = tail[cursor]
        cursor += 1
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        flag, sep, inline = token.partition("=")
        name = _name(flag)
        if name in RESUME_NAMES:
            return finish(reason="resume_precedence_not_verified_in_code")
        if name not in (MERGE_NAMES if merge else WEIGHT_NAMES):
            continue
        if sep:
            values = [inline]
        elif merge:
            values = []
            while cursor < len(tail) and not tail[cursor].startswith("-"):
                values.append(tail[cursor])
                cursor += 1
        else:
            values = tail[cursor : cursor + 1]
            cursor += 1
        if not values or any(v.startswith("--") for v in values):
            return finish(reason="missing_checkpoint_option_value")
        for value in values:
            declared.append(
                (
                    value.split(",") if merge else value,
                    None,
                    "explicit_argv_without_verified_parser:" + name,
                )
            )
    if declared:
        return finish(declared)
    parent = (setup.get("parent_checkpoint") or {}).get("path")
    if parent and not merge:
        return finish([(parent, None, "structured_parent_path_no_explicit_override")])
    return finish(reason="no_statically_resolved_checkpoint_input")
