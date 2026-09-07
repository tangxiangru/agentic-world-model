"""Frozen-cohort TRAIN-only operational-feature experiment; no full-text approval.

Run prepare to freeze features/spec/source hashes, then fit exactly once into
that new artifact. Only whitelisted declared operands or AST literal operands
reach features. Syntax presence is not execution or artifact-fidelity proof.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import shlex
from pathlib import Path

import numpy as np

from tools.outcome_prediction import wm_parameter_pilot as baseline

DATASETS = {
    "gsm8k": ("openai/gsm8k",),
    "openmathinstruct2": ("nvidia/openmathinstruct-2", "openmathinstruct-2"),
    "metamathqa": ("meta-math/metamathqa", "metamathqa"),
    "orca_math": ("microsoft/orca-math-word-problems-200k",),
    "openr1_math": ("open-r1/openr1-math-220k",),
    "mixture_of_thoughts": ("open-r1/mixture-of-thoughts",),
    "openmathreasoning": ("nvidia/openmathreasoning",),
}
METHODS = {"sft", "lora", "qlora", "dpo", "grpo", "rft", "merge", "decoding", "evaluation"}
PEFT = {"none", "lora", "qlora", "adalora", "ia3"}
OPTIMIZERS = {
    "adamw_torch",
    "adamw_torch_fused",
    "adamw_8bit",
    "paged_adamw_8bit",
    "paged_adamw_32bit",
    "adamw_bnb_8bit",
    "adamw",
    "adam",
    "sgd",
    "adafactor",
}
ENUMS = {
    "optim": OPTIMIZERS,
    "optimizer": OPTIMIZERS,
    "attn_implementation": {"flash_attention_2", "sdpa", "eager"},
    "lr_scheduler_type": set(baseline.CATEGORIES["scheduler"]),
    "save_strategy": {"no", "steps", "epoch", "best"},
    "loss_type": {"sigmoid", "hinge", "ipo", "kto_pair", "robust", "sft"},
    "bias": {"none", "all", "lora_only"},
    "task_type": {"causal_lm", "seq_2_seq_lm"},
    "reduction": {"mean", "sum", "none"},
    "dtype": {"bfloat16", "float16", "float32", "bf16", "fp16", "fp32"},
    "torch_dtype": {"bfloat16", "float16", "float32", "bf16", "fp16", "fp32"},
}
NUMERIC_OPERANDS = {
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_length",
    "max_seq_length",
    "warmup_ratio",
    "warmup_steps",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "max_grad_norm",
    "r",
    "lora_alpha",
    "lora_dropout",
    "beta",
    "label_smoothing_factor",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "min_p",
}
BOOL_OPERANDS = {
    "packing",
    "padding_free",
    "completion_only_loss",
    "assistant_only_loss",
    "gradient_checkpointing",
    "bf16",
    "fp16",
    "group_by_length",
    "use_cache",
    "use_reentrant",
    "load_in_4bit",
    "load_in_8bit",
    "use_rslora",
    "use_dora",
    "add_special_tokens",
    "truncation",
    "do_sample",
    "enable_thinking",
}
CALLS = {
    "TrainingArguments",
    "SFTConfig",
    "DPOConfig",
    "GRPOConfig",
    "LoraConfig",
    "AdaLoraConfig",
    "IA3Config",
    "BitsAndBytesConfig",
    "GenerationConfig",
    "SFTTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    "Trainer",
    "AdamW",
    "AdamW8bit",
    "PagedAdamW8bit",
    "Adam",
    "SGD",
    "Adafactor",
    "CrossEntropyLoss",
    "cross_entropy",
    "get_peft_model",
    "prepare_model_for_kbit_training",
    "apply_chat_template",
    "gradient_checkpointing_enable",
    "merge_and_unload",
}
FLAG_NUMBERS = {
    "--lr",
    "--learning-rate",
    "--epochs",
    "--num-train-epochs",
    "--bs",
    "--batch-size",
    "--grad-accum",
    "--accum",
    "--gradient-accumulation-steps",
    "--max-len",
    "--maxlen",
    "--max-length",
    "--max-seq-length",
    "--warmup",
    "--weight-decay",
    "--lora-r",
    "--lora-rank",
    "--lora-alpha",
    "--lora-dropout",
    "--beta",
    "--max-new-tokens",
    "--max-tokens",
    "--temperature",
    "--top-p",
    "--top-k",
    "--min-p",
    "--repetition-penalty",
}
FLAG_ENUMS = {
    "--optim": OPTIMIZERS,
    "--optimizer": OPTIMIZERS,
    "--attn-implementation": ENUMS["attn_implementation"],
    "--precision": set(baseline.CATEGORIES["precision"]),
    "--scheduler": set(baseline.CATEGORIES["scheduler"]),
}
FLAG_BOOLS = {"--pack", "--packing", "--lora", "--qlora", "--no-gc", "--bf16", "--fp16"}
QUOTA_FLAGS = {
    "--max-examples",
    "--max-samples",
    "--num-examples",
    "--num-samples",
    "--n-examples",
    "--n-samples",
    "--limit",
}
TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "all-linear",
}
RICH_VARIANTS = ("rich_ridge_full", "rich_extra_trees_full", "rich_extra_trees_last_step")
SPEC = {
    "scope": "exploratory_same106_train_only_operational_feature_pilot",
    "primary_comparison": "rich_extra_trees_full versus saved extra_trees_full",
    "primary_metrics": ["selected_final_accuracy", "selection_regret"],
    "selection_population": "only same-run groups with at least two frozen cohort targets",
    "secondary_metric": "equal_run_mae",
    "models": list(RICH_VARIANTS),
    "hyperparameters": copy.deepcopy(baseline.SPEC),
    "baselines": "reuse all four verified frozen parameter_pilot_v1 OOF arms without refitting",
    "folds": "exact saved parameter_pilot_v1 fold identities and ordering",
    "new_features": "canonical external sources and explicit build-command caps; exact method/PEFT categories; whitelisted command operands; known constructor/call presence and literal typed operands",
    "no_features": [
        "free prose",
        "comments",
        "docstrings",
        "logging strings",
        "arbitrary identifiers or constants",
        "n_examples",
        "realized yield",
        "progress counts",
        "ancestor labels",
        "test inputs",
        "file paths",
        "source hashes",
    ],
    "code_semantics": "literal syntax only, not execution: no variable resolution, runtime evaluation, effective-value precedence or scanner-derived approval",
    "quota_policy": "only positive literal known cap flags from simple declared builder argv for a single canonical external dataset; never n_examples or inferred output cardinality",
    "no_tuning_after_fit": True,
    "no_final_model_or_rpm_claim": True,
}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def _jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.writelines(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        )
    Path(path).chmod(0o600)


def _literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in {str, bool, int, float}:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        if type(value) in {int, float}:
            return -value if isinstance(node.op, ast.USub) else value
    return None


def _name(node):
    return (
        node.id
        if isinstance(node, ast.Name)
        else node.attr
        if isinstance(node, ast.Attribute)
        else ""
    )


def _tokens(command):
    if isinstance(command, str):
        try:
            command = shlex.split(command)
        except ValueError:
            return []
    if not isinstance(command, list) or not all(isinstance(t, str) for t in command):
        return []
    if any(t in {";", "&&", "||", "|", ">", ">>"} or "$" in t or "`" in t for t in command):
        return []
    return command


def _flag_values(command):
    tokens = _tokens(command)
    for i, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        flag, separator, value = token.partition("=")
        flag = flag.replace("_", "-")
        if not separator:
            value = tokens[i + 1] if i + 1 < len(tokens) else None
        yield flag, value


def step_features(row):
    """No access to labels, prior history, problem, hypothesis, results or conclusion."""
    source = row["model_input"]
    setup = source["plan"].get("setup", {})
    features, proofs, omitted = {}, [], []

    def emit(key, value, path, kind):
        features[key] = value
        proofs.append({"feature": key, "source_path": path, "extraction": kind})

    method = setup.get("method", {})
    if isinstance(method, dict):
        for field, allowed in (("family", METHODS), ("peft", PEFT)):
            value = method.get(field)
            if isinstance(value, str) and value.strip().lower() in allowed:
                emit(
                    "declared." + field,
                    value.strip().lower(),
                    "/plan/setup/method/" + field,
                    "exact_enum",
                )
            elif value is not None:
                omitted.append(
                    {"path": "/plan/setup/method/" + field, "reason": "not_exact_whitelisted_enum"}
                )
    data = setup.get("data", [])
    for i, entry in enumerate(data if isinstance(data, list) else []):
        if not isinstance(entry, dict):
            continue
        raw_source = entry.get("source")
        names = [
            name
            for name, aliases in DATASETS.items()
            if isinstance(raw_source, str) and any(a in raw_source.lower() for a in aliases)
        ]
        safe = isinstance(raw_source, str) and not baseline.SELF_DATA.search(raw_source)
        if not safe or not names:
            omitted.append(
                {"path": f"/plan/setup/data/{i}", "reason": "not_unambiguous_external_source"}
            )
            continue
        for name in names:
            emit(
                f"data[{i}].source.{name}",
                1.0,
                f"/plan/setup/data/{i}/source",
                "canonical_external_dataset",
            )
        # Only a single external source may receive a declared command cap.
        builder_tokens = _tokens(entry.get("build_command"))
        built_by = entry.get("built_by")
        direct_builder = (
            len(builder_tokens) >= 2
            and Path(builder_tokens[0]).name.startswith("python")
            and isinstance(built_by, str)
            and Path(builder_tokens[1]).name == Path(built_by).name
            and not any(token.startswith("#") for token in builder_tokens)
        )
        if len(names) == 1 and direct_builder:
            for flag, value in _flag_values(builder_tokens):
                number = baseline._scalar(value)
                if flag in QUOTA_FLAGS and number is not None and number > 0:
                    emit(
                        f"data[{i}].planned_cap.{flag}",
                        number,
                        f"/plan/setup/data/{i}/build_command",
                        "explicit_external_builder_cap",
                    )
        if "n_examples" in entry:
            omitted.append(
                {
                    "path": f"/plan/setup/data/{i}/n_examples",
                    "reason": "cardinality_not_assumed_planned_quota",
                }
            )
    command = setup.get("command", {})
    command = command.get("argv") if isinstance(command, dict) else command
    for flag, value in _flag_values(command):
        if flag in FLAG_BOOLS:
            emit("command." + flag, 1.0, "/plan/setup/command/argv", "literal_flag_presence")
        elif flag in FLAG_NUMBERS:
            number = baseline._scalar(value)
            if number is not None:
                emit(
                    "command." + flag,
                    number,
                    "/plan/setup/command/argv",
                    "whitelisted_numeric_flag",
                )
        elif flag in FLAG_ENUMS and isinstance(value, str) and value.lower() in FLAG_ENUMS[flag]:
            emit(
                "command." + flag,
                value.lower(),
                "/plan/setup/command/argv",
                "whitelisted_enum_flag",
            )
    code_audit = []
    for index, code in enumerate(source.get("code", [])):
        if (
            not isinstance(code, dict)
            or code.get("status") != "reconstructed"
            or not isinstance(code.get("content"), str)
        ):
            continue
        content = code["content"]
        role = code.get("role", "")
        if role != "training" and not role.startswith("data_builder_"):
            omitted.append(
                {"path": f"/code/{index}", "reason": "non_training_or_builder_code_not_admitted"}
            )
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            omitted.append({"path": f"/code/{index}", "reason": "unparseable_python"})
            continue
        code_audit.append(
            {"index": index, "role": role, "sha256": hashlib.sha256(content.encode()).hexdigest()}
        )

        class Visitor(ast.NodeVisitor):
            def __init__(self, code_role, code_index):
                self.code_role = code_role
                self.code_index = code_index

            def visit_Expr(self, node):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return  # Discard every string expression, including all docstrings.
                self.generic_visit(node)

            def visit_Call(self, node):
                call = _name(node.func)
                if call in {
                    "print",
                    "info",
                    "debug",
                    "warning",
                    "warn",
                    "error",
                    "exception",
                    "log",
                }:
                    return  # Do not mine nested expressions in logging arguments either.
                prefix = f"syntax.{self.code_role}.{call}"
                path = f"/code/{self.code_index}/content:line={node.lineno}"
                if call in CALLS:
                    emit(prefix + ".present", 1.0, path, "known_forward_call_syntax_presence")
                    for keyword in node.keywords:
                        key, value = keyword.arg, _literal(keyword.value)
                        if (
                            key in NUMERIC_OPERANDS
                            and type(value) in {int, float}
                            and math.isfinite(value)
                        ):
                            # Separate literal-value indicators avoid claiming effective precedence.
                            emit(
                                f"{prefix}.{key}.literal={value}",
                                1.0,
                                path,
                                "literal_numeric_keyword_syntax",
                            )
                        elif key in BOOL_OPERANDS and type(value) is bool:
                            emit(
                                f"{prefix}.{key}.literal={value}",
                                1.0,
                                path,
                                "literal_boolean_keyword_syntax",
                            )
                        elif (
                            key in ENUMS and isinstance(value, str) and value.lower() in ENUMS[key]
                        ):
                            emit(
                                f"{prefix}.{key}.literal={value.lower()}",
                                1.0,
                                path,
                                "literal_whitelisted_enum_keyword_syntax",
                            )
                        elif key == "target_modules" and isinstance(
                            keyword.value, (ast.List, ast.Tuple)
                        ):
                            values = [_literal(n) for n in keyword.value.elts]
                            if all(isinstance(v, str) and v in TARGET_MODULES for v in values):
                                for value in values:
                                    emit(
                                        f"{prefix}.target_module={value}",
                                        1.0,
                                        path,
                                        "literal_lora_projection_name",
                                    )
                self.generic_visit(node)

        Visitor(role, index).visit(tree)
    return features, {
        "feature_provenance": proofs,
        "omissions": omitted,
        "code_sources": code_audit,
    }


def richer_cohort(frozen, rows):
    lookup = {row["example_id"]: row for row in rows}
    needed = sorted({identity for item in frozen for identity in item["closure_ids_private"]})
    features, audit = {}, {}
    for identity in needed:
        row = lookup[identity]
        if baseline._data_dependency_screen(row):
            raise ValueError(
                "Frozen closure contains unresolved/self-generated data; do not relax cohort"
            )
        features[identity], audit[identity] = step_features(row)
    output = []
    for item in frozen:
        rich = copy.deepcopy(item)
        rich["features"] = copy.deepcopy(item["features"])
        for index, identity in enumerate(item["closure_ids_private"]):
            rich["features"].update(
                {f"step_{index}.rich.{key}": value for key, value in features[identity].items()}
            )
        rich["last_step_features"] = {
            **item["last_step_features"],
            **{"rich." + k: v for k, v in features[item["example_id"]].items()},
        }
        output.append(rich)
    return output, audit


def prepare_artifact(baseline_dir, inventory_path, split_path, graph_path, output_dir):
    output, previous = Path(output_dir), Path(baseline_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a NEW immutable prepared pilot directory")
    verification = json.loads((previous / "verification.json").read_text())
    for name, expected in verification["artifact_sha256"].items():
        if _hash(previous / name) != expected:
            raise ValueError("Frozen parameter-pilot artifact changed")
    split = json.loads(Path(split_path).read_text())
    rows, skipped = baseline.load_train_inventory(inventory_path, split)
    graph = json.loads(Path(graph_path).read_text())
    reproduced, omitted = baseline.prepare_cohort(rows, split, graph)
    frozen = [json.loads(line) for line in (previous / "cohort.jsonl").read_text().splitlines()]
    if baseline.digest(reproduced) != baseline.digest(frozen) or len(frozen) != 106:
        raise ValueError("The exact frozen 106-example cohort cannot be reproduced")
    richer, audits = richer_cohort(frozen, rows)
    folds = json.loads((previous / "folds.json").read_text())
    output.mkdir(parents=True, mode=0o700)
    _json(output / "spec.json", SPEC)
    _jsonl(output / "cohort.jsonl", richer)
    _json(output / "feature_audit.json", audits)
    _json(
        output / "folds.json",
        [
            {
                k: f[k]
                for k in (
                    "benchmark",
                    "fold",
                    "train_cells",
                    "validation_cells",
                    "train_examples",
                    "validation_examples",
                )
            }
            for f in folds
        ],
    )
    _json(
        output / "coverage.json",
        {
            "retained_recipes": len(richer),
            "retained_runs": len({r["cell_id"] for r in richer}),
            "omitted_train_cards": omitted,
            "test_lines_skipped_before_decode": sum(skipped.values()),
            "same_original_cohort_and_closures": True,
        },
    )
    _json(
        output / "freeze.json",
        {
            "scope": SPEC["scope"],
            "status": "features_and_model_spec_frozen_before_fit",
            "source_sha256": _hash(__file__),
            "baseline_source_sha256": _hash(baseline.__file__),
            "source_inputs": {
                str(path): _hash(path)
                for path in (
                    Path(inventory_path),
                    Path(split_path),
                    Path(graph_path),
                    previous / "cohort.jsonl",
                    previous / "folds.json",
                    previous / "predictions.jsonl",
                )
            },
            "baseline_dir": str(previous),
            "baseline_predictions_sha256": _hash(previous / "predictions.jsonl"),
            "files_sha256": {p.name: _hash(p) for p in sorted(output.iterdir())},
            "fit_has_not_run": True,
            "heldout_labels_used": False,
        },
    )
    return {
        "frozen_recipes": len(richer),
        "unique_closure_steps": len(audits),
        "new_feature_keys": len({k for item in richer for k in item["features"] if ".rich." in k}),
    }


def fit_artifact(output_dir):
    output = Path(output_dir)
    if (output / "fit_started.json").exists():
        raise FileExistsError("This frozen artifact has already been attempted; no reruns")
    freeze = json.loads((output / "freeze.json").read_text())
    if (
        _hash(__file__) != freeze["source_sha256"]
        or _hash(baseline.__file__) != freeze["baseline_source_sha256"]
    ):
        raise ValueError("Pilot source changed after feature/spec freeze")
    for name, expected in freeze["files_sha256"].items():
        if _hash(output / name) != expected:
            raise ValueError("Prepared features/spec/folds changed after freeze")
    for path, expected in freeze["source_inputs"].items():
        if _hash(path) != expected:
            raise ValueError("Original input artifact changed after freeze")
    _json(
        output / "fit_started.json",
        {"freeze_sha256": _hash(output / "freeze.json"), "one_fit_attempt": True},
    )
    rows = [json.loads(line) for line in (output / "cohort.jsonl").read_text().splitlines()]
    by_id = {r["example_id"]: r for r in rows}
    folds = json.loads((output / "folds.json").read_text())
    previous = Path(freeze["baseline_dir"])
    predictions = [
        json.loads(line) for line in (previous / "predictions.jsonl").read_text().splitlines()
    ]
    pred_by_id = {p["example_id"]: p for p in predictions}
    if set(pred_by_id) != set(by_id):
        raise ValueError("Baseline prediction cohort differs")
    for identity, row in by_id.items():
        if any(pred_by_id[identity][k] != row[k] for k in ("cell_id", "benchmark", "target")):
            raise ValueError("Baseline target or identity changed")
    fold_audits = []
    for fold in folds:
        training = [by_id[k] for k in fold["train_examples"]]
        validation = [by_id[k] for k in fold["validation_examples"]]
        if {r["cell_id"] for r in training} & {r["cell_id"] for r in validation}:
            raise ValueError("TRAIN CV run overlap")
        for variant in RICH_VARIANTS:
            base_variant = variant.removeprefix("rich_")
            outputs, audit = baseline._model_predict(training, validation, base_variant)
            if not all(np.isfinite(outputs)):
                raise ValueError("Nonfinite predictions; do not change settings and retry")
            fold_audits.append(
                {
                    "benchmark": fold["benchmark"],
                    "fold": fold["fold"],
                    "variant": variant,
                    "fit_examples": fold["train_examples"],
                    "validation_examples": fold["validation_examples"],
                    "fold_train_features": audit["feature_names"],
                }
            )
            for row, prediction in zip(validation, outputs, strict=True):
                original = pred_by_id[row["example_id"]]
                if original["fold"] != fold["fold"] or original["benchmark"] != fold["benchmark"]:
                    raise ValueError("Original CV assignment differs")
                original["predictions"][variant] = float(prediction)
    names = sorted(predictions[0]["predictions"])
    metrics = {
        benchmark: {
            variant: baseline._metrics(
                [r for r in predictions if r["benchmark"] == benchmark], variant
            )
            for variant in names
        }
        for benchmark in sorted({r["benchmark"] for r in rows})
    }
    _jsonl(output / "predictions.jsonl", sorted(predictions, key=lambda r: r["example_id"]))
    _json(output / "fold_feature_audit.json", fold_audits)
    _json(
        output / "report.json",
        {
            "scope": SPEC["scope"],
            "spec": SPEC,
            "metrics": metrics,
            "fit_scope": "TRAIN-only CV on same frozen 106 recipes; no new model selected or served",
            "no_post_result_feature_or_model_tuning": True,
            "heldout_labels_used": False,
            "heldout_scored": False,
            "limits": [
                "Limited operational features, not an approved full recipe or replacement selector input.",
                "Adaptive historical within-run candidates are not simultaneous prospective proposals.",
                "Literal syntax features are not effective configuration or executed checkpoint identity.",
                "Missing/unwhitelisted features are omitted; neither scanner clearance nor narrative summaries imply code semantics.",
            ],
        },
    )
    _json(
        output / "fit_provenance.json",
        {
            "freeze_sha256": _hash(output / "freeze.json"),
            "source_sha256": _hash(__file__),
            "files_sha256": {p.name: _hash(p) for p in sorted(output.iterdir())},
            "sources_unchanged": _hash(__file__) == freeze["source_sha256"]
            and _hash(baseline.__file__) == freeze["baseline_source_sha256"],
        },
    )
    return {
        benchmark: {
            variant: {
                key: values[key]
                for key in (
                    "selected_final_accuracy",
                    "selection_regret",
                    "equal_run_mae",
                    "selection_runs_with_two_or_more_candidates",
                )
            }
            for variant, values in group.items()
        }
        for benchmark, group in metrics.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for flag in ("baseline-dir", "inventory", "split", "graph", "output-dir"):
        prep.add_argument("--" + flag, type=Path, required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = (
        fit_artifact(args.output_dir)
        if args.command == "fit"
        else prepare_artifact(
            args.baseline_dir, args.inventory, args.split, args.graph, args.output_dir
        )
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
