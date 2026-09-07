"""Bounded, non-executing extraction of pre-proposal Python configuration.

Only the planned training entrypoint is inspected. No archived imports, calls,
filesystem access, environment reads, eval, or exec are performed. A small AST
interpreter propagates literal settings through arithmetic, dictionaries,
argparse and explicitly invoked local functions. Unknown control flow remains
unknown. These are statically supported *configuration declarations*, not a
guarantee about the runtime behavior of arbitrary Python or library defaults.

The caller owns timestamp/first-proposal eligibility. This module must receive
the same pre-proposal reconstructed ``model_input`` used by that gate.
"""

from __future__ import annotations

import ast
import hashlib
import math
import posixpath
import re
import shlex
from collections import defaultdict
from dataclasses import dataclass, field


class _Unknown:
    def __deepcopy__(self, memo):
        return self


UNKNOWN = _Unknown()

ALIASES = {
    "learning_rate": ("learning_rate", "lr"),
    "epochs": ("num_train_epochs", "epochs"),
    "batch_size": ("per_device_train_batch_size", "batch_size", "bs", "micro_bs", "per_device_batch"),
    "grad_accum": ("gradient_accumulation_steps", "grad_accum", "accum", "ga"),
    "sequence_length": ("max_seq_length", "max_seq_len", "max_length", "max_len", "maxlen", "block_size"),
    "max_steps": ("max_steps",),
    "warmup_ratio": ("warmup_ratio", "warmup"),
    "warmup_steps": ("warmup_steps",),
    "weight_decay": ("weight_decay", "wd"),
    "beta1": ("adam_beta1", "beta1"),
    "beta2": ("adam_beta2", "beta2"),
    "epsilon": ("adam_epsilon", "eps", "epsilon"),
    "grad_clip": ("max_grad_norm", "grad_clip", "clip_grad_norm"),
    "min_lr_ratio": ("min_lr_ratio", "min_lr_rate"),
    "min_lr": ("min_lr", "eta_min"),
    "lora_rank": ("lora_r", "lora_rank"),
    "lora_alpha": ("lora_alpha",),
    "lora_dropout": ("lora_dropout",),
    "max_new_tokens": ("max_new_tokens", "max_tokens", "max_completion_length"),
    "max_prompt_length": ("max_prompt_length",),
    "temperature": ("temperature", "temp"),
    "top_p": ("top_p",),
    "top_k": ("top_k",),
    "min_p": ("min_p",),
    "repetition_penalty": ("repetition_penalty",),
    "num_generations": ("num_generations",),
    "kl_beta": ("beta", "kl_beta"),
    "completion_only": ("completion_only_loss", "completion_only"),
    "assistant_only": ("assistant_only_loss", "assistant_only"),
    "packing": ("packing",),
    "padding_free": ("padding_free",),
    "gradient_checkpointing": ("gradient_checkpointing", "grad_ckpt", "grad_ckpt_enabled"),
    "do_sample": ("do_sample",),
    "group_by_length": ("group_by_length",),
    "use_liger": ("use_liger_kernel", "liger"),
    "optimizer": ("optim", "optimizer"),
    "scheduler": ("lr_scheduler_type", "scheduler"),
    "weight_precision": ("torch_dtype", "load_dtype"),
    "compute_precision": ("bnb_4bit_compute_dtype",),
}
ALIAS_TO_FIELD = {alias: name for name, aliases in ALIASES.items() for alias in aliases}
NUMERIC = tuple(list(ALIASES)[:27])
FLAGS = (
    "completion_only", "assistant_only", "packing", "padding_free", "gradient_checkpointing",
    "do_sample", "group_by_length", "use_liger", "eos_override", "pad_override",
    "lora_present", "load_in_4bit", "load_in_8bit", "optimizer_fused",
    "label_mask_syntax",
)
CATEGORIES = {
    "optimizer": ("adamw", "adam", "sgd", "adafactor", "other"),
    "scheduler": ("cosine", "cosine_with_min_lr", "linear", "constant", "constant_with_warmup", "other"),
    "weight_precision": ("bf16", "fp16", "fp32"),
    "compute_precision": ("bf16", "fp16", "fp32"),
}
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "all_linear")
FEATURE_KEYS = tuple(
    ["codecfg." + name for name in (*NUMERIC, *FLAGS)]
    + [f"codecfg.{name}.{category}" for name, categories in CATEGORIES.items() for category in categories]
    + ["codecfg.lora_target." + target for target in LORA_TARGETS]
    + ["codecfg.effective_batch_per_device", "codecfg.lora_alpha_per_rank", "codecfg.entrypoint_available"]
)
TRAIN_CONFIGS = {"TrainingArguments", "SFTConfig", "GRPOConfig", "DPOConfig", "PPOConfig"}
DECODE_CONFIGS = {"SamplingParams", "GenerationConfig"}
OPTIMIZERS = {"AdamW": "adamw", "Adam": "adam", "SGD": "sgd", "Adafactor": "adafactor"}


@dataclass(frozen=True)
class _Symbol:
    name: str


@dataclass
class _Parser:
    arguments: list = field(default_factory=list)
    defaults: dict = field(default_factory=dict)
    unresolved: bool = False


@dataclass
class _Config:
    kind: str
    values: dict


class _ReturnFlow(Exception):
    def __init__(self, value):
        self.value = value


class _StopFlow(Exception):
    pass


def _finite(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.strip()):
            return None
        value = float(value)
    if not isinstance(value, (float, int)) or abs(value) > 1e12 or not math.isfinite(value):
        return None
    return float(value)


def _precision(value):
    if isinstance(value, _Symbol):
        value = value.name
    if not isinstance(value, str):
        return UNKNOWN
    terms = re.findall(r"(?<!\w)(?:bfloat16|bf16|float16|fp16|float32|fp32)(?!\w)", value.lower())
    precision_names = {"bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "fp16": "fp16", "float32": "fp32", "fp32": "fp32"}
    normalized = {precision_names[v] for v in terms}
    return next(iter(normalized)) if len(normalized) == 1 else UNKNOWN


def _category(name, value):
    if name.endswith("precision"):
        return _precision(value)
    if not isinstance(value, str):
        return UNKNOWN
    value = value.lower().strip().replace("-", "_")
    if name == "optimizer":
        return next((v for v in ("adamw", "adafactor", "adam", "sgd") if value == v or value.startswith(v + "_")), "other")
    return value if value in CATEGORIES[name] else "other"


def _normalized(field_name, value):
    if value is UNKNOWN or value is None:
        return UNKNOWN
    if field_name in CATEGORIES:
        return _category(field_name, value)
    if field_name in FLAGS or field_name.startswith("lora_target."):
        if isinstance(value, bool) or (isinstance(value, (float, int)) and value in (0, 1)):
            return float(value)
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return float(value.lower() == "true")
        return UNKNOWN
    number = _finite(value)
    if number is None or (number < 0 and field_name not in {"max_steps", "top_k"}):
        return UNKNOWN
    return number


def _command_tokens(command):
    argv = command.get("argv", [])
    if isinstance(argv, str):
        try:
            argv = shlex.split(argv)
        except ValueError:
            return []
    return argv if isinstance(argv, list) and all(isinstance(v, str) for v in argv) else []


def _canonical_path(path, cwd):
    if not isinstance(path, str) or not path:
        return None
    return posixpath.normpath(path if path.startswith("/") else posixpath.join(cwd, path))


def _script_index(argv):
    """Recognize direct Python and a small explicit set of Python launchers.

    Shells, env wrappers, expansion/substitution and compound commands are
    refused. This is command-shape validation, not execution certification.
    """
    if not argv or any(re.search(r"[;&|<>`\n\r]|\$[({]", token) for token in argv):
        return None
    launcher = posixpath.basename(argv[0])
    index = 1
    distributed = launcher == "torchrun"
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", launcher):
        while index < len(argv) and argv[index] in {"-u", "-B", "-O", "-OO", "-I", "-s", "-S", "-E"}:
            index += 1
        if argv[index:index + 2] == ["-m", "torch.distributed.run"]:
            distributed = True
            index += 2
        elif index < len(argv) and argv[index].startswith("-"):
            return None
    elif launcher == "accelerate" and argv[1:2] == ["launch"]:
        distributed = True
        index = 2
    elif not distributed:
        return None
    if distributed:
        value_options = {
            "--nproc_per_node", "--nproc-per-node", "--nnodes", "--node_rank", "--node-rank",
            "--master_addr", "--master-addr", "--master_port", "--master-port", "--rdzv_backend",
            "--rdzv-backend", "--rdzv_endpoint", "--rdzv-endpoint", "--rdzv_id", "--rdzv-id",
            "--num_processes", "--num_machines", "--machine_rank", "--mixed_precision", "--config_file",
        }
        flags = {"--standalone", "--multi_gpu", "--cpu"}
        while index < len(argv) and argv[index].startswith("-"):
            flag, sep, _ = argv[index].partition("=")
            if flag in flags and not sep:
                index += 1
            elif flag in value_options:
                index += 1 if sep else 2
            else:
                return None
    return index if index < len(argv) and argv[index].endswith(".py") else None


def _literal_known(value):
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return _finite(value) is not None
    if isinstance(value, (list, tuple)):
        return all(_literal_known(v) for v in value)
    if isinstance(value, dict):
        return all(_literal_known(k) and _literal_known(v) for k, v in value.items())
    return False


def _select_entrypoint(model_input):
    setup = (model_input.get("plan") or {}).get("setup") or {}
    command = setup.get("command") or {}
    if not isinstance(command, dict):
        command = {}
    argv = _command_tokens(command)
    cwd = command.get("cwd") if isinstance(command.get("cwd"), str) else ""
    script = _canonical_path(command.get("script"), cwd)
    script_index = _script_index(argv)
    if script_index is None:
        return None, [], "unsupported_or_ambiguous_launcher"
    invoked_script = _canonical_path(argv[script_index], cwd)
    if script and invoked_script != script:
        return None, [], "command_script_conflict"
    script = script or invoked_script
    if not script:
        return None, [], "missing_planned_entrypoint"
    matching = [c for c in model_input.get("code", []) if isinstance(c, dict) and _canonical_path(c.get("script_path"), cwd) == script]
    usable = [c for c in matching if c.get("status") == "reconstructed" and isinstance(c.get("content"), str)]
    contents = {c["content"] for c in usable}
    if len(contents) != 1:
        return None, [], "missing_or_conflicting_reconstructed_entrypoint"
    # Only arguments after the invoked script belong to its argparse parser.
    return usable[0], argv[script_index + 1:], None


class _Interpreter:
    """Partial AST semantics; unsupported expressions are opaque unknowns."""

    def __init__(self, argv):
        self.argv = argv
        self.evidence = defaultdict(list)
        self.audit = {"unsupported_control_flow": 0, "unsupported_expressions": 0, "parsed_argument_count": 0, "configuration_calls": 0}
        self.depth = 0

    def record(self, name, value, priority=3, source="configuration"):
        if name in NUMERIC or name in FLAGS or name in CATEGORIES or name.startswith("lora_target."):
            self.evidence[name].append((priority, _normalized(name, value), source))

    def settings(self, kind, values, priority=3, source="configuration"):
        allowed = set(ALIAS_TO_FIELD)
        if kind in DECODE_CONFIGS or kind == "generate":
            allowed = {"max_new_tokens", "max_tokens", "max_length", "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "do_sample"}
        for key, value in values.items():
            if key in allowed:
                name = ALIAS_TO_FIELD[key]
                if key == "max_length" and (kind in DECODE_CONFIGS or kind == "generate"):
                    # Total-length generation cap is not a new-token cap.
                    continue
                self.record(name, value, priority, source)
        for key, name in (("bf16", "bf16"), ("fp16", "fp16")):
            if values.get(key) is True:
                self.record("compute_precision", name, priority, source)
            elif values.get(key) is UNKNOWN:
                self.record("compute_precision", UNKNOWN, priority, source)
        if kind == "from_pretrained":
            for key in ("torch_dtype", "dtype"):
                if key in values:
                    self.record("weight_precision", values[key], priority, source)
        if kind == "autocast" and "dtype" in values:
            self.record("compute_precision", values["dtype"], priority, source)
        for key, flag in (("eos_token_id", "eos_override"), ("eos_token", "eos_override"), ("stop_token_ids", "eos_override"), ("pad_token_id", "pad_override"), ("pad_token", "pad_override")):
            if key in values:
                self.record(flag, values[key] is not None, priority, source)
        for key in ("load_in_4bit", "load_in_8bit"):
            if key in values:
                self.record(key, values[key], priority, source)
        scheduler = values.get("lr_scheduler_kwargs")
        if isinstance(scheduler, dict):
            for name in ("min_lr_ratio", "min_lr_rate", "min_lr"):
                if name in scheduler:
                    self.record(ALIAS_TO_FIELD[name], scheduler[name], priority, source)
        if kind == "LoraConfig":
            self.record("lora_present", True, priority, source)
            if "r" in values:
                self.record("lora_rank", values["r"], priority, source)
            targets = values.get("target_modules", UNKNOWN)
            if isinstance(targets, str):
                targets = [targets]
            if isinstance(targets, (list, tuple)) and all(isinstance(v, str) for v in targets):
                targets = {v.replace("-", "_") for v in targets}
                for target in LORA_TARGETS:
                    self.record("lora_target." + target, target in targets, priority, source)
        if kind in OPTIMIZERS:
            self.record("optimizer", OPTIMIZERS[kind], priority, source)
            betas = values.get("betas")
            if isinstance(betas, (list, tuple)) and len(betas) == 2:
                self.record("beta1", betas[0], priority, source)
                self.record("beta2", betas[1], priority, source)
            if "fused" in values:
                self.record("optimizer_fused", values["fused"], priority, source)

    def value(self, node, env):
        if node is None:
            return UNKNOWN
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, (str, int, float, bool, type(None))) else UNKNOWN
        if isinstance(node, ast.Name):
            return env.get(node.id, _Symbol(node.id) if node.id in {"int", "float", "str", "bool", "dict", "min", "max", "round", "len"} else UNKNOWN)
        if isinstance(node, ast.Attribute):
            base = self.value(node.value, env)
            if isinstance(base, dict):
                return base.get(node.attr, UNKNOWN)
            if isinstance(base, _Config):
                return base.values.get(node.attr, UNKNOWN)
            if isinstance(base, _Symbol):
                return _Symbol(base.name + "." + node.attr)
            return UNKNOWN
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            result = [self.value(n, env) for n in node.elts]
            return tuple(result) if isinstance(node, ast.Tuple) else result
        if isinstance(node, ast.Dict):
            result = {}
            for key, val in zip(node.keys, node.values):
                value = self.value(val, env)
                if key is None:
                    if not isinstance(value, dict):
                        return UNKNOWN
                    result.update(value)
                else:
                    key = self.value(key, env)
                    if not isinstance(key, (str, int, float, bool)):
                        return UNKNOWN
                    result[key] = value
            return result
        if isinstance(node, ast.Subscript):
            base, key = self.value(node.value, env), self.value(node.slice, env)
            if isinstance(base, (dict, list, tuple)) and isinstance(key, (str, int)):
                try:
                    return base[key]
                except (KeyError, IndexError, TypeError):
                    pass
            return UNKNOWN
        if isinstance(node, ast.UnaryOp):
            value = self.value(node.operand, env)
            if isinstance(node.op, ast.Not) and _literal_known(value):
                return not bool(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return -value if isinstance(node.op, ast.USub) else value if isinstance(node.op, ast.UAdd) else UNKNOWN
            return UNKNOWN
        if isinstance(node, ast.BinOp):
            left, right = self.value(node.left, env), self.value(node.right, env)
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and _finite(v) is not None for v in (left, right)):
                return UNKNOWN
            try:
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                elif isinstance(node.op, ast.Div):
                    result = left / right
                elif isinstance(node.op, ast.FloorDiv):
                    result = left // right
                else:
                    return UNKNOWN
                return result if _finite(result) is not None else UNKNOWN
            except (ArithmeticError, OverflowError):
                return UNKNOWN
        if isinstance(node, ast.IfExp):
            test = self.value(node.test, env)
            if not _literal_known(test):
                return UNKNOWN
            return self.value(node.body if bool(test) else node.orelse, env)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.value(node.left, env), self.value(node.comparators[0], env)
            if not _literal_known(left) or not _literal_known(right):
                return UNKNOWN
            op = node.ops[0]
            if isinstance(op, (ast.Eq, ast.Is)):
                return left == right
            if isinstance(op, (ast.NotEq, ast.IsNot)):
                return left != right
            if isinstance(left, (float, int)) and isinstance(right, (float, int)):
                if isinstance(op, ast.Gt):
                    return left > right
                if isinstance(op, ast.GtE):
                    return left >= right
                if isinstance(op, ast.Lt):
                    return left < right
                if isinstance(op, ast.LtE):
                    return left <= right
            return UNKNOWN
        if isinstance(node, ast.BoolOp):
            value = UNKNOWN
            for operand in node.values:
                value = self.value(operand, env)
                if not _literal_known(value):
                    return UNKNOWN
                if isinstance(node.op, ast.And) and not bool(value):
                    return value
                if isinstance(node.op, ast.Or) and bool(value):
                    return value
            return value
        if isinstance(node, ast.Call):
            return self.call(node, env)
        self.audit["unsupported_expressions"] += 1
        return UNKNOWN

    def keyword_values(self, node, env):
        values = {}
        for kw in node.keywords:
            value = self.value(kw.value, env)
            if kw.arg is None:
                if isinstance(value, dict):
                    values.update(value)
                else:
                    # An opaque kwargs mapping might override any setting.
                    values.update({alias: UNKNOWN for alias in ALIAS_TO_FIELD})
            else:
                values[kw.arg] = value
        return values

    def parse_arguments(self, parser):
        result = {}
        sources = {}
        events = []

        def convert(value, spec):
            type_value = spec.get("type")
            converter = type_value.name if isinstance(type_value, _Symbol) else None
            if isinstance(value, str) and converter in {"int", "float", "str", "bool"}:
                try:
                    return {"int": int, "float": float, "str": str, "bool": bool}[converter](value)
                except (ValueError, OverflowError):
                    return UNKNOWN
            if type_value is not None and converter not in {"int", "float", "str", "bool"}:
                return UNKNOWN
            return value

        for flags, spec in parser.arguments:
            dest = spec.get("dest") or next((s[2:] for s in flags if s.startswith("--")), flags[0].lstrip("-"))
            if not isinstance(dest, str):
                continue
            dest = dest.replace("-", "_")
            action = spec.get("action", "store")
            value = spec.get("default", False if action == "store_true" else True if action == "store_false" else None)
            if dest not in result:
                result[dest] = convert(parser.defaults.get(dest, value), spec)
                sources[dest] = "argparse_default"
            for i, token in enumerate(self.argv):
                if token == "--":
                    break
                flag, sep, inline = token.partition("=")
                if flag not in flags:
                    continue
                if action in {"store_true", "store_false"}:
                    value = action == "store_true"
                elif action == "store_const":
                    value = spec.get("const", UNKNOWN)
                elif action != "store" or "nargs" in spec:
                    value = UNKNOWN
                else:
                    value = inline if sep else self.argv[i + 1] if i + 1 < len(self.argv) else UNKNOWN
                events.append((i, dest, convert(value, spec)))
            self.audit["parsed_argument_count"] += 1
        for _, dest, value in sorted(events, key=lambda item: item[0]):
            result[dest] = value
            sources[dest] = "parsed_argv"
        if parser.unresolved:
            result = dict.fromkeys(result, UNKNOWN)
        for dest, value in result.items():
            if dest in ALIAS_TO_FIELD:
                self.record(ALIAS_TO_FIELD[dest], value, 2, sources[dest])
        return result

    def call(self, node, env):
        func = self.value(node.func, env)
        base = func.name.split(".")[-1] if isinstance(func, _Symbol) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        owner = self.value(node.func.value, env) if isinstance(node.func, ast.Attribute) else None
        values = self.keyword_values(node, env)
        args = [self.value(a, env) for a in node.args]
        if base == "ArgumentParser":
            return _Parser()
        if isinstance(owner, _Parser) and base == "add_argument":
            flags = [a for a in args if isinstance(a, str)]
            if flags:
                owner.arguments.append((flags, values))
            return UNKNOWN
        if isinstance(owner, _Parser) and base == "set_defaults":
            owner.defaults.update(values)
            return None
        if isinstance(owner, _Parser) and base in {"parse_args", "parse_known_args"}:
            # Explicit alternate argv (e.g. parse_args([])) changes semantics.
            if args or values:
                return UNKNOWN
            result = self.parse_arguments(owner)
            return (result, []) if base == "parse_known_args" else result
        if isinstance(owner, dict) and base == "update":
            if args and isinstance(args[0], dict):
                owner.update(args[0])
            owner.update(values)
            return None
        if isinstance(owner, dict) and base == "get" and args:
            return owner.get(args[0], args[1] if len(args) > 1 else None) if isinstance(args[0], (str, int)) else UNKNOWN
        if base in TRAIN_CONFIGS | DECODE_CONFIGS | set(OPTIMIZERS) | {"LoraConfig", "BitsAndBytesConfig", "from_pretrained", "autocast", "generate"}:
            self.settings(base, values)
            self.audit["configuration_calls"] += 1
            return _Config(base, values) if base in TRAIN_CONFIGS | DECODE_CONFIGS | {"LoraConfig", "BitsAndBytesConfig"} else UNKNOWN
        if base in {"train_on_responses_only", "DataCollatorForCompletionOnlyLM"}:
            self.record("completion_only", True)
        if base == "gradient_checkpointing_enable":
            self.record("gradient_checkpointing", True)
        if base == "clip_grad_norm_" and len(args) >= 2:
            self.record("grad_clip", args[1])
        if base == "dict":
            return {**(args[0] if args and isinstance(args[0], dict) else {}), **values}
        if base in {"float", "int", "bool", "str"} and len(args) == 1 and isinstance(args[0], (str, bool, int, float)):
            try:
                return {"float": float, "int": int, "bool": bool, "str": str}[base](args[0])
            except (ValueError, OverflowError):
                return UNKNOWN
        if base in {"min", "max", "round"} and args and all(_finite(v) is not None for v in args):
            try:
                return {"min": min, "max": max, "round": round}[base](*args)
            except (TypeError, ValueError, ArithmeticError):
                return UNKNOWN
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and self.depth < 4:
            if func.args.vararg or func.args.kwarg or func.args.kwonlyargs:
                return UNKNOWN
            local = dict(env)
            defaults = [UNKNOWN] * (len(func.args.args) - len(func.args.defaults)) + [self.value(d, env) for d in func.args.defaults]
            for index, (arg, default) in enumerate(zip(func.args.args, defaults)):
                local[arg.arg] = args[index] if index < len(args) else values.get(arg.arg, default)
            self.depth += 1
            try:
                result = self.block(func.body, local)
            except _ReturnFlow as control:
                result = control.value
            finally:
                self.depth -= 1
            return result
        return UNKNOWN

    def assign(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            vals = value if isinstance(value, (list, tuple)) and len(value) == len(target.elts) else [UNKNOWN] * len(target.elts)
            for name, val in zip(target.elts, vals):
                self.assign(name, val, env)
        elif isinstance(target, ast.Attribute):
            owner = self.value(target.value, env)
            if isinstance(owner, dict):
                owner[target.attr] = value
            if isinstance(owner, _Config):
                owner.values[target.attr] = value
                self.settings(owner.kind, {target.attr: value})
            # Named generation-config assignments are allowed, never token IDs.
            if isinstance(target.value, ast.Attribute) and target.value.attr == "generation_config":
                self.settings("GenerationConfig", {target.attr: value})
        elif isinstance(target, ast.Subscript):
            owner, key = self.value(target.value, env), self.value(target.slice, env)
            if isinstance(owner, dict) and isinstance(key, (str, int)):
                owner[key] = value

    def uncertain(self, nodes, env):
        self.audit["unsupported_control_flow"] += 1
        for statement in nodes:
            for node in ast.walk(statement):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                        self.assign(target, UNKNOWN, env)
                if isinstance(node, ast.Call):
                    func = self.value(node.func, env)
                    base = func.name.split(".")[-1] if isinstance(func, _Symbol) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                    owner = self.value(node.func.value, env) if isinstance(node.func, ast.Attribute) else None
                    if isinstance(owner, _Parser):
                        if base == "set_defaults" and all(kw.arg is not None for kw in node.keywords):
                            owner.defaults.update({kw.arg: UNKNOWN for kw in node.keywords})
                        else:
                            owner.unresolved = True
                    if base in TRAIN_CONFIGS | DECODE_CONFIGS | set(OPTIMIZERS) | {"LoraConfig", "BitsAndBytesConfig", "from_pretrained", "autocast", "generate"}:
                        previous_counts = {name: len(items) for name, items in self.evidence.items()}
                        self.settings(base, {kw.arg: UNKNOWN for kw in node.keywords if kw.arg})
                        for name, items in self.evidence.items():
                            for index in range(previous_counts.get(name, 0), len(items)):
                                priority, _, _ = items[index]
                                items[index] = (priority, UNKNOWN, "uncertain_control_flow")

    def block(self, nodes, env):
        for index, node in enumerate(nodes):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    env[alias.asname or alias.name.split(".")[0]] = _Symbol(alias.name if alias.asname else alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    env[alias.asname or alias.name] = _Symbol((node.module or "") + "." + alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                env[node.name] = node
            elif isinstance(node, ast.Assign):
                value = self.value(node.value, env)
                for target in node.targets:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AnnAssign):
                self.assign(node.target, self.value(node.value, env), env)
            elif isinstance(node, ast.AugAssign):
                self.assign(node.target, UNKNOWN, env)
            elif isinstance(node, ast.Expr):
                self.value(node.value, env)
            elif isinstance(node, ast.If):
                test = self.value(node.test, env)
                if not _literal_known(test):
                    self.uncertain(node.body + node.orelse, env)
                    if any(isinstance(n, (ast.Return, ast.Raise, ast.Break, ast.Continue)) for branch in node.body + node.orelse for n in ast.walk(branch)):
                        self.uncertain(nodes[index + 1:], env)
                        raise _StopFlow()
                else:
                    self.block(node.body if bool(test) else node.orelse, env)
            elif isinstance(node, ast.Return):
                raise _ReturnFlow(self.value(node.value, env))
            elif isinstance(node, (ast.Raise, ast.Break, ast.Continue)):
                raise _StopFlow()
            elif isinstance(node, ast.Assert):
                test = self.value(node.test, env)
                if not _literal_known(test):
                    self.uncertain(nodes[index + 1:], env)
                    raise _StopFlow()
                if not bool(test):
                    raise _StopFlow()
            elif isinstance(node, (ast.For, ast.While, ast.Try, ast.TryStar, ast.With, ast.AsyncWith, ast.Match)):
                self.uncertain([node], env)
        return UNKNOWN


def _has_label_mask_syntax(tree):
    """Syntax evidence only: a label assignment contains an explicit -100.

    Includes uncalled functions; deliberately does NOT imply an active
    completion-only loss, a particular mask boundary, or runtime execution.
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        label_target = any(
            (isinstance(part, ast.Name) and part.id in {"labels", "label_ids"})
            or (isinstance(part, ast.Attribute) and part.attr in {"labels", "label_ids"})
            or (isinstance(part, ast.Subscript) and isinstance(part.slice, ast.Constant) and part.slice.value in {"labels", "label_ids"})
            for target in targets for part in ast.walk(target)
        )
        if label_target and node.value is not None and any(
            isinstance(part, ast.UnaryOp) and isinstance(part.op, ast.USub)
            and isinstance(part.operand, ast.Constant) and part.operand.value == 100
            for part in ast.walk(node.value)
        ):
            return True
    return False


def extract_config_features(model_input: dict) -> tuple[dict[str, float | None], dict]:
    """Return fixed numeric features and a separate extraction/provenance audit.

    Precedence: supported config use > parsed planned argv/default > structured
    first-plan hyperparameter. Conflicting or unknown strongest evidence is
    missing, not averaged, backfilled from a stale plan, or treated as zero.
    """
    interpreter = _Interpreter([])
    setup = (model_input.get("plan") or {}).get("setup") or {}
    method = setup.get("method") or {}
    hp = method.get("hyperparams") or {}
    if isinstance(hp, dict):
        interpreter.settings("planned", hp, 1, "planned_hyperparams")
        if "precision" in hp:
            interpreter.record("compute_precision", hp["precision"], 1, "planned_hyperparams")
    code, argv, reason = _select_entrypoint(model_input)
    parsed = False
    if code is not None:
        try:
            tree = ast.parse(code["content"])
            if sum(1 for _ in ast.walk(tree)) > 100_000:
                reason = "entrypoint_ast_too_large"
            else:
                interpreter.argv = argv
                try:
                    interpreter.block(tree.body, {"__name__": "__main__"})
                except (_ReturnFlow, _StopFlow):
                    pass
                interpreter.record("label_mask_syntax", _has_label_mask_syntax(tree), source="syntax_evidence_not_execution")
                parsed = True
        except (SyntaxError, ValueError, RecursionError):
            reason = "entrypoint_parse_failed"
    result = dict.fromkeys(FEATURE_KEYS)
    audit = dict(interpreter.audit)
    audit.update({"entrypoint_available": parsed, "entrypoint_reason": reason, "entrypoint_sha256": hashlib.sha256(code["content"].encode()).hexdigest() if code else None, "field_sources": {}, "conflicting_fields": [], "unknown_fields": [], "runtime_certified": False})
    for name, evidence in interpreter.evidence.items():
        priority = max(v[0] for v in evidence)
        strongest = [v for v in evidence if v[0] == priority]
        values = [v[1] for v in strongest]
        known = {v for v in values if v is not UNKNOWN}
        audit["field_sources"][name] = sorted({v[2] for v in strongest})
        if any(v is UNKNOWN for v in values) or len(known) != 1:
            audit["unknown_fields"].append(name)
            if len(known) > 1:
                audit["conflicting_fields"].append(name)
            continue
        value = next(iter(known))
        if name in CATEGORIES:
            for category in CATEGORIES[name]:
                result[f"codecfg.{name}.{category}"] = float(value == category)
        else:
            result["codecfg." + name] = float(value)
    for output, a, b, divide in (
        ("effective_batch_per_device", "batch_size", "grad_accum", False),
        ("lora_alpha_per_rank", "lora_alpha", "lora_rank", True),
    ):
        left, right = result["codecfg." + a], result["codecfg." + b]
        if left is not None and right is not None and right > 0:
            result["codecfg." + output] = left / right if divide else left * right
    result["codecfg.entrypoint_available"] = float(parsed)
    audit["conflicting_fields"].sort()
    audit["unknown_fields"].sort()
    return result, audit
