"""Static, score-blind evidence for planned checkpoint output paths.

This resolves a *path binding*, not checkpoint bytes, successful execution,
training completion, or grading provenance. The caller must independently
corroborate which archived checkpoint was graded and apply the other clean-row
checks. In particular a mutable final snapshot is never a source of proof.

For differing paths, only a pre-proposal reconstructed declared entrypoint is
interpreted. The bounded interpreter never executes source/imports or accesses
the filesystem/environment. It supports literal paths, argparse, os.path.join,
Path / components, formatting, and explicitly invoked local helper functions.
The terminal weight-bearing save must be unconditional and resolve exactly to
the target. Earlier periodic saves can precede that terminal save; a later
unknown/alternative save or weight change prevents an alias proof.

``audit.final_output_checkpoint`` is an optional caller-supplied corroborated
archive-source override. Its presence is authoritative even when None; absent
that field, the legacy final-output comparison supplies the proposed target.
Neither source alone is certified by this component.
The ``reconstructed`` source status is also a caller-supplied provenance
assertion: this component does not independently verify trace cutoffs.
"""

from __future__ import annotations

import ast
import hashlib
import posixpath
from dataclasses import dataclass, field
from datetime import datetime

from tools.outcome_prediction.wm_checkpoint_inputs import _parse_arguments
from tools.outcome_prediction.wm_code_features import (
    TRAIN_CONFIGS,
    UNKNOWN,
    _Config,
    _Interpreter,
    _ReturnFlow,
    _select_entrypoint,
    _StopFlow,
    _Symbol,
)

MAX_PATH_LENGTH = 4096
MODEL_CLASSES = {
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForSeq2SeqLM",
    "Gemma3ForConditionalGeneration",
    "Qwen3ForCausalLM",
    "PeftModel",
    "AutoPeftModelForCausalLM",
}
TRAINER_CLASSES = {"Trainer", "SFTTrainer", "GRPOTrainer", "DPOTrainer", "PPOTrainer"}
AUXILIARY_CLASSES = {"AutoTokenizer", "AutoProcessor", "GenerationConfig", "AutoConfig"}
SAVE_METHODS = {"save_pretrained", "save_model", "save_checkpoint", "_save_checkpoint"}
PATH_MUTATIONS = {
    "os.rename",
    "os.replace",
    "os.symlink",
    "os.link",
    "os.remove",
    "os.unlink",
    "shutil.move",
    "shutil.copy",
    "shutil.copy2",
    "shutil.copytree",
    "shutil.rmtree",
}
PATH_MUTATION_NAMES = {name.rsplit(".", 1)[-1] for name in PATH_MUTATIONS} | {
    "rmdir",
    "symlink_to",
    "hardlink_to",
}


class _Path(str):
    """Marks pathlib semantics; ordinary Python strings do not support /."""


@dataclass
class _Artifact:
    kind: str
    properties: dict = field(default_factory=dict)


@dataclass
class _BoundSave:
    owner: _Artifact
    method: str


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _normalize(path, cwd):
    if not isinstance(path, str) or not path.strip() or len(path) > MAX_PATH_LENGTH:
        return None
    if any(token in path for token in ("\x00", "$", "~", "\n", "\r")):
        return None
    if not posixpath.isabs(path):
        if (
            not isinstance(cwd, str)
            or not posixpath.isabs(cwd)
            or any(t in cwd for t in ("$", "~", "\x00"))
        ):
            return None
        path = posixpath.join(cwd, path)
    return posixpath.normpath(path)


class _PathInterpreter(_Interpreter):
    def __init__(self, argv, cwd):
        super().__init__(argv)
        self.cwd = cwd
        self.events = []
        self.trainer_classes = set()
        self.path_operations = []
        self.opaque_helpers = []
        self.validation_errors = []
        self.class_methods_overridden = set()
        self.symbol_overrides = set()

    def parse_arguments(self, parser):
        # Stricter than the feature extractor: invalid/unknown argv must not
        # leave a seemingly valid output default behind. parse_known_args is
        # intentionally treated just as strictly in this bounded proof.
        values, _, reason = _parse_arguments(parser.arguments, parser.defaults, self.argv)
        if parser.unresolved:
            reason = "conditional_argument_declarations"
        if reason:
            self.validation_errors.append(reason)
            return dict.fromkeys(values, UNKNOWN)
        return values

    def checked_keywords(self, node, env):
        """Expand only statically known kwargs, rejecting Python-invalid duplicates."""
        result = {}
        for keyword in node.keywords:
            value = self.value(keyword.value, env)
            entries = value if keyword.arg is None else {keyword.arg: value}
            if not isinstance(entries, dict) or any(not isinstance(k, str) for k in entries):
                return None
            if result.keys() & entries.keys():
                return None
            result.update(entries)
        return result

    def value(self, node, env):
        if isinstance(node, ast.Attribute):
            owner = self.value(node.value, env)
            if isinstance(owner, _Symbol) and owner.name + "." + node.attr in self.symbol_overrides:
                return UNKNOWN
            if isinstance(owner, _Artifact):
                if node.attr in owner.properties:
                    return owner.properties[node.attr]
                if node.attr in SAVE_METHODS and owner.kind in {"model", "trainer"}:
                    return _BoundSave(owner, node.attr)
                if node.attr == "model" and owner.kind == "trainer":
                    return _Artifact("model")
                if node.attr in {"config", "generation_config", "tokenizer", "processing_class"}:
                    return _Artifact("auxiliary")
                return UNKNOWN
        if isinstance(node, ast.BinOp):
            left, right = self.value(node.left, env), self.value(node.right, env)
            if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
                combined = left + right
                return combined if len(combined) <= MAX_PATH_LENGTH else UNKNOWN
            if isinstance(node.op, ast.Div) and isinstance(left, _Path) and isinstance(right, str):
                return _Path(posixpath.join(left, right))
        if isinstance(node, ast.JoinedStr):
            parts = []
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    parts.append(part.value)
                elif (
                    isinstance(part, ast.FormattedValue)
                    and part.format_spec is None
                    and part.conversion in (-1, 115)
                ):
                    value = self.value(part.value, env)
                    if not isinstance(value, (str, int)) or isinstance(value, bool):
                        return UNKNOWN
                    parts.append(str(value))
                else:
                    return UNKNOWN
            value = "".join(parts)
            return value if len(value) <= MAX_PATH_LENGTH else UNKNOWN
        return super().value(node, env)

    def assign(self, target, value, env):
        if isinstance(target, ast.Attribute):
            owner = self.value(target.value, env)
            if isinstance(owner, _Symbol):
                self.symbol_overrides.add(owner.name + "." + target.attr)
            if target.attr in SAVE_METHODS:
                self.event(target, "save_method_override", receiver="declared_monkeypatch")
                if isinstance(owner, _Symbol):
                    # Inheritance/aliasing is opaque; any class-level save
                    # override disables subsequent proof using that method.
                    self.class_methods_overridden.add(target.attr)
            if isinstance(owner, _Artifact):
                owner.properties[target.attr] = value
                return
        super().assign(target, value, env)

    def event(self, node, kind, path=None, unconditional=True, receiver=None):
        self.events.append(
            {
                "kind": kind,
                "line": node.lineno,
                "unconditional": unconditional,
                "path": _normalize(path, self.cwd),
                "receiver_kind": receiver,
                "expression": ast.unparse(node)[:700],
            }
        )

    def call(self, node, env):
        func = self.value(node.func, env)
        qualified = func.name if isinstance(func, _Symbol) else ""
        base = (
            qualified.split(".")[-1]
            if qualified
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        owner = self.value(node.func.value, env) if isinstance(node.func, ast.Attribute) else None
        if isinstance(func, _BoundSave) and base not in SAVE_METHODS:
            # Alias calls are recognized as possible weight writes, but are
            # deliberately not an additional positive-proof syntax.
            self.event(node, "weight_save", receiver="aliased_save_method")
            return UNKNOWN
        if qualified in {
            "torch.save",
            "safetensors.torch.save_file",
            "safetensors.torch.save_model",
        }:
            # Raw serialization may write model weights outside save_pretrained;
            # it must not silently disappear after the apparent terminal save.
            self.event(node, "weight_save", receiver="raw_weight_serialization")
            return UNKNOWN
        if qualified in {
            "os.path.join",
            "posixpath.join",
            "os.path.normpath",
            "posixpath.normpath",
            "os.path.abspath",
        }:
            args = [self.value(a, env) for a in node.args]
            if not args or node.keywords or not all(isinstance(a, str) for a in args):
                return UNKNOWN
            if base == "join":
                value = posixpath.join(*args)
            elif len(args) == 1:
                value = (
                    _normalize(args[0], self.cwd)
                    if base == "abspath"
                    else posixpath.normpath(args[0])
                )
            else:
                return UNKNOWN
            return value if isinstance(value, str) and len(value) <= MAX_PATH_LENGTH else UNKNOWN
        if qualified in {"pathlib.Path", "pathlib.PurePath", "pathlib.PurePosixPath"}:
            args = [self.value(a, env) for a in node.args]
            if args and all(isinstance(a, str) for a in args) and not node.keywords:
                return _Path(posixpath.join(*args))
            return UNKNOWN
        if isinstance(owner, _Path) and base == "joinpath":
            args = [self.value(a, env) for a in node.args]
            return (
                _Path(posixpath.join(owner, *args))
                if all(isinstance(a, str) for a in args) and not node.keywords
                else UNKNOWN
            )
        if qualified == "os.chdir":
            args = [self.value(a, env) for a in node.args]
            self.cwd = (
                _normalize(args[0], self.cwd) if len(args) == 1 and not node.keywords else None
            )
            self.path_operations.append(
                {"line": node.lineno, "operation": "declared_chdir", "cwd": self.cwd}
            )
            return None
        if qualified in PATH_MUTATIONS or (
            isinstance(owner, _Path) and base in PATH_MUTATION_NAMES
        ):
            self.event(node, "artifact_path_change", receiver="declared_filesystem_operation")
            return UNKNOWN
        if base == "from_pretrained" and qualified.startswith(("transformers.", "peft.")):
            class_name = qualified.split(".")[-2]
            if class_name in MODEL_CLASSES:
                return _Artifact("model")
            if class_name in AUXILIARY_CLASSES:
                return _Artifact("auxiliary")
        if (
            base in TRAINER_CLASSES and qualified.startswith(("transformers.", "trl."))
        ) or qualified in self.trainer_classes:
            values = self.checked_keywords(node, env)
            if values is None:
                return UNKNOWN
            return _Artifact(
                "trainer",
                {
                    "model": values.get("model", _Artifact("model")),
                    "args": values.get("args", UNKNOWN),
                },
            )
        if qualified == "peft.get_peft_model":
            args = [self.value(a, env) for a in node.args]
            return (
                _Artifact("model")
                if args and isinstance(args[0], _Artifact) and args[0].kind == "model"
                else UNKNOWN
            )
        if isinstance(owner, _Artifact) and base == "merge_and_unload" and owner.kind == "model":
            self.event(node, "weight_change", receiver="planned_lora_merge")
            return _Artifact("model")
        if isinstance(owner, _Artifact) and base in {"train", "fit"} and owner.kind == "trainer":
            self.event(node, "weight_change", receiver="trainer")
            return UNKNOWN
        if isinstance(node.func, ast.Attribute) and base in SAVE_METHODS:
            if isinstance(owner, _Artifact) and owner.kind == "auxiliary":
                return None
            if isinstance(owner, _Config) and owner.kind in {
                "GenerationConfig",
                "BitsAndBytesConfig",
            }:
                return None
            args = [self.value(a, env) for a in node.args]
            values = self.checked_keywords(node, env)
            weight_receiver = (
                isinstance(owner, _Artifact)
                and (
                    (owner.kind == "model" and base == "save_pretrained")
                    or (owner.kind == "trainer" and base == "save_model")
                )
                and base not in owner.properties
                and base not in self.class_methods_overridden
            )
            destination = "save_directory" if base == "save_pretrained" else "output_dir"
            valid_signature = (
                values is not None
                and len(args) <= 1
                and not (args and destination in values)
                and not ({"output_dir", "save_directory"} - {destination}) & values.keys()
            )
            values = values or {}
            path = args[0] if args else values.get(destination, UNKNOWN)
            if not args and not values and base == "save_model" and isinstance(owner, _Artifact):
                config = owner.properties.get("args")
                if isinstance(config, _Config):
                    path = config.values.get("output_dir", UNKNOWN)
            self.event(
                node,
                "weight_save",
                path if weight_receiver and valid_signature else UNKNOWN,
                receiver=owner.kind if isinstance(owner, _Artifact) else "unknown",
            )
            return None
        if base in TRAIN_CONFIGS and qualified.startswith(("transformers.", "trl.")):
            values = self.checked_keywords(node, env)
            if values is None or (node.args and "output_dir" in values):
                return UNKNOWN
            if node.args and "output_dir" not in values:
                values["output_dir"] = self.value(node.args[0], env)
            return _Config(base, values)
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The inherited feature interpreter deliberately approximates
            # helper binding. Path proofs additionally require ordinary,
            # nondecorated synchronous calls with unambiguous Python arity.
            values = self.checked_keywords(node, env)
            names = [arg.arg for arg in func.args.args]
            supplied_positionally = set(names[: len(node.args)])
            required = set(names[: len(names) - len(func.args.defaults)])
            if (
                isinstance(func, ast.AsyncFunctionDef)
                or func.decorator_list
                or func.args.posonlyargs
                or func.args.vararg
                or func.args.kwarg
                or func.args.kwonlyargs
                or any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(func))
                or self.depth >= 4
                or values is None
                or len(node.args) > len(names)
                or any(isinstance(arg, ast.Starred) for arg in node.args)
                or bool(set(values or {}) - set(names))
                or bool(supplied_positionally & set(values or {}))
                or bool(required - supplied_positionally - set(values or {}))
            ):
                self.validation_errors.append("unsupported_or_ambiguous_local_helper_call")
                return UNKNOWN
        try:
            return super().call(node, env)
        except _StopFlow:
            # Dataset/loss helpers commonly return under runtime conditions.
            # Their return value stays unknown, like an imported opaque helper;
            # it must not abort reasoning about independent later path literals.
            # Potential checkpoint effects remain conditional so a helper after
            # the apparent final save cannot silently hide later selection.
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and func.name != "main":
                self.opaque_helpers.append({"line": node.lineno, "function": func.name})
                for possible in ast.walk(func):
                    if isinstance(possible, ast.Call):
                        possible_func = self.value(possible.func, env)
                        full = possible_func.name if isinstance(possible_func, _Symbol) else ""
                        if full in PATH_MUTATIONS:
                            self.event(
                                possible,
                                "artifact_path_change",
                                unconditional=False,
                                receiver="opaque_local_helper",
                            )
                            continue
                    if isinstance(possible, ast.Call) and isinstance(possible.func, ast.Attribute):
                        if possible.func.attr in SAVE_METHODS:
                            self.event(
                                possible,
                                "weight_save",
                                unconditional=False,
                                receiver="opaque_local_helper",
                            )
                        elif possible.func.attr in {"train", "fit", "merge_and_unload"}:
                            self.event(
                                possible,
                                "weight_change",
                                unconditional=False,
                                receiver="opaque_local_helper",
                            )
                        elif possible.func.attr in PATH_MUTATION_NAMES:
                            self.event(
                                possible,
                                "artifact_path_change",
                                unconditional=False,
                                receiver="opaque_local_helper",
                            )
                return UNKNOWN
            raise

    def uncertain(self, nodes, env):
        # Unknown earlier checkpoint saves are allowed to be dominated by a
        # later unconditional terminal save. Unknown LATER saves are ambiguous.
        # Imports under try are not executed by the abstract interpreter. An
        # unshadowed AutoProcessor/Tokenizer import still establishes that its
        # own save call cannot write model weights, whether the import succeeds
        # or the exception skips that call entirely.
        auxiliary_imports = set()
        assigned = set()
        for statement in nodes:
            for node in ast.walk(statement):
                if isinstance(node, ast.ImportFrom) and node.module == "transformers":
                    auxiliary_imports.update(
                        alias.asname or alias.name
                        for alias in node.names
                        if alias.name in AUXILIARY_CLASSES
                    )
                elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    assigned.update(
                        part.id
                        for target in targets
                        for part in ast.walk(target)
                        if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
                    )
        auxiliary_imports -= assigned
        for statement in nodes:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                func = self.value(node.func, env)
                full = func.name if isinstance(func, _Symbol) else ""
                if isinstance(func, _BoundSave) or full in {
                    "torch.save",
                    "safetensors.torch.save_file",
                    "safetensors.torch.save_model",
                }:
                    self.event(
                        node,
                        "weight_save",
                        unconditional=False,
                        receiver="uncertain_alias_or_raw_serialization",
                    )
                    continue
                if full in PATH_MUTATIONS:
                    self.event(
                        node,
                        "artifact_path_change",
                        unconditional=False,
                        receiver="uncertain_branch",
                    )
                    continue
                if not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr in SAVE_METHODS:
                    receiver = node.func.value
                    if (
                        isinstance(receiver, ast.Call)
                        and isinstance(receiver.func, ast.Attribute)
                        and receiver.func.attr == "from_pretrained"
                        and isinstance(receiver.func.value, ast.Name)
                        and receiver.func.value.id in auxiliary_imports
                    ):
                        continue
                    owner = self.value(node.func.value, env)
                    if isinstance(owner, _Artifact) and owner.kind == "auxiliary":
                        continue
                    if isinstance(owner, _Config) and owner.kind in {
                        "GenerationConfig",
                        "BitsAndBytesConfig",
                    }:
                        continue
                    self.event(
                        node, "weight_save", unconditional=False, receiver="uncertain_branch"
                    )
                elif node.func.attr == "chdir":
                    self.cwd = None
                elif node.func.attr in PATH_MUTATION_NAMES:
                    self.event(
                        node,
                        "artifact_path_change",
                        unconditional=False,
                        receiver="uncertain_branch",
                    )
                elif node.func.attr in {"train", "fit", "merge_and_unload"}:
                    owner = self.value(node.func.value, env)
                    if isinstance(owner, _Artifact) and owner.kind in {"model", "trainer"}:
                        self.event(
                            node, "weight_change", unconditional=False, receiver="uncertain_branch"
                        )
        super().uncertain(nodes, env)

    def block(self, nodes, env):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                bases = [self.value(base, env) for base in node.bases]
                known = all(
                    isinstance(base, _Symbol)
                    and (
                        base.name in self.trainer_classes
                        or base.name.startswith(("transformers.", "trl."))
                        and base.name.split(".")[-1] in TRAINER_CLASSES
                    )
                    for base in bases
                )
                overrides = {
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                overrides.update(
                    target.id
                    for member in node.body
                    if isinstance(member, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                    for assigned in (
                        member.targets if isinstance(member, ast.Assign) else [member.target]
                    )
                    for target in ast.walk(assigned)
                    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)
                )
                if (
                    bases
                    and known
                    and not node.decorator_list
                    and not overrides.intersection(
                        SAVE_METHODS | {"__init__", "__getattribute__", "__getattr__"}
                    )
                ):
                    name = "local_trainer." + node.name
                    self.trainer_classes.add(name)
                    env[node.name] = _Symbol(name)
                else:
                    env[node.name] = UNKNOWN
            else:
                super().block([node], env)
        return UNKNOWN


def resolve_target_binding(row):
    """Resolve known paths or a statically supported terminal planned save.

    Never reads label values, final result prose, snapshots, or predictions.
    Caller-supplied archive corroboration and remaining clean criteria stay
    separate; status is only a path-binding decision.
    """
    audit = _mapping(row.get("audit"))
    source = _mapping(row.get("model_input"))
    setup = _mapping(_mapping(source.get("plan")).get("setup"))
    command = _mapping(setup.get("command"))
    cwd = command.get("cwd")
    comparison = _mapping(audit.get("output_artifact_comparison"))
    declared = setup.get("output_dir", comparison.get("first_declared_output_dir"))
    overridden = "final_output_checkpoint" in audit
    target = (
        audit.get("final_output_checkpoint")
        if overridden
        else comparison.get("final_output_checkpoint")
    )
    planned_path, target_path = _normalize(declared, cwd), _normalize(target, cwd)
    result = {
        "status": "unresolved",
        "planned_output": planned_path,
        "target_output": target_path,
        "target_source": "caller_corroborated_archive_override"
        if overridden
        else "final_card_comparison_unverified",
        "evidence": [],
        "candidate_save_paths": [],
        "reasons": [],
        "runtime_certified": False,
        "checkpoint_bytes_certified": False,
        "score_values_read": False,
        "scope": "Path binding only; caller must verify official archive provenance and other clean-row criteria",
    }
    if planned_path is None or target_path is None:
        result["reasons"].append("unknown_planned_or_target_path")
        return result
    audit_declared = comparison.get("first_declared_output_dir")
    if audit_declared is not None and _normalize(audit_declared, cwd) != planned_path:
        result["reasons"].append("conflicting_planned_output_paths")
        return result
    if planned_path == target_path:
        result["status"] = "exact"
        result["evidence"].append(
            {
                "kind": "lexical_path_identity",
                "path": target_path,
                "normalization": "POSIX lexical only; no symlink or checkpoint-prefix collapse",
            }
        )
        return result
    structural_reasons = audit.get("reasons")
    if not isinstance(structural_reasons, list):
        result["reasons"].append("unknown_first_record_audit")
        return result
    if row.get("first_stage") != "plan" or "first_result_not_empty" in structural_reasons:
        result["reasons"].append("no_prospective_first_plan")
        return result
    try:
        at = datetime.fromisoformat(row["first_submitted_at"].replace("Z", "+00:00"))
        if at.tzinfo is None:
            raise ValueError("Naive proposal timestamp")
    except (KeyError, TypeError, ValueError, AttributeError):
        result["reasons"].append("invalid_proposal_timestamp")
        return result
    code, argv, reason = _select_entrypoint(source)
    if code is None:
        result["reasons"].append(reason or "missing_reconstructed_entrypoint")
        return result
    interpreter = _PathInterpreter(argv, cwd)
    try:
        tree = ast.parse(code["content"])
        if sum(1 for _ in ast.walk(tree)) > 100_000:
            raise ValueError("Entrypoint AST too large")
        interpreter.block(tree.body, {"__name__": "__main__"})
    except (_ReturnFlow, _StopFlow):
        result["reasons"].append("unresolved_program_termination")
    except (SyntaxError, ValueError, RecursionError, TypeError, AttributeError, IndexError):
        result["reasons"].append("unsupported_or_invalid_entrypoint")
    digest = hashlib.sha256(code["content"].encode()).hexdigest()
    result["evidence"] = [{**event, "script_sha256": digest} for event in interpreter.events]
    result["opaque_helpers"] = interpreter.opaque_helpers
    result["reasons"].extend(interpreter.validation_errors)
    result["candidate_save_paths"] = sorted(
        {
            event["path"]
            for event in interpreter.events
            if event["kind"] == "weight_save" and event["path"]
        }
    )
    if result["reasons"]:
        return result
    if not interpreter.events:
        result["reasons"].append("no_proven_model_save")
        return result
    terminal = interpreter.events[-1]
    if terminal["kind"] != "weight_save":
        result["reasons"].append("weight_change_after_last_save")
    elif not terminal["unconditional"] or terminal["path"] is None:
        result["reasons"].append("terminal_model_save_ambiguous")
    elif terminal["path"] != target_path:
        result["reasons"].append("target_is_not_terminal_planned_model_save")
    else:
        result["status"] = "planned_save_path"
    return result
