"""PRIVATE TRAIN-only limited-parameter pilot, not an approved full-recipe WM.

No serving, held-out evaluation, text/code features, API calls, or artifact writes.
The frozen TRAIN partition is checked before target labels are accessed. Every
prefix is an independent target provisionally; no ancestor scores enter features.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SEED = 20260905
NUMERIC_KEYS = {
    "lr",
    "epochs",
    "batch_size",
    "grad_accum",
    "max_seq_len",
    "warmup",
    "weight_decay",
    "seed",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "min_p",
}
CATEGORIES = {
    "precision": {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "float32": "fp32",
        "fp16": "fp16",
        "float16": "fp16",
    },
    "scheduler": {
        k: k
        for k in (
            "cosine",
            "cosine_with_min_lr",
            "constant",
            "constant_with_warmup",
            "linear",
            "none",
        )
    },
}
ROLES = {"training", "merge", "decoding", "evaluation"}
BASES = {"google/gemma-3-4b-pt", "Qwen/Qwen3-4B-Base"}
EXTERNAL_DATASETS = {
    "openai/gsm8k",
    "nvidia/openmathinstruct-2",
    "meta-math/metamathqa",
    "metamathqa",
    "openmathinstruct-2",
    "microsoft/orca-math-word-problems-200k",
    "open-r1/openr1-math-220k",
    "open-r1/mixture-of-thoughts",
    "nvidia/openmathreasoning",
}
SELF_DATA = re.compile(
    r"synthetic\s*:\s*self|self[- ]generated|\brft\b|rejection[- ]sampl|\bexp[-_ ]?\d+|\b(?:ckpts|sft_out)[/_-]",
    re.IGNORECASE,
)
SPEC = {
    "status": "exploratory_private_limited_feature_cohort_pilot",
    "seed": SEED,
    "folds": 4,
    "ridge": {"alpha": 10.0, "solver": "lsqr"},
    "extra_trees": {"n_estimators": 300, "min_samples_leaf": 3, "max_features": 1.0, "n_jobs": 1},
    "training_weight": "equal total weight per TRAIN run within each fold",
    "target": "target recipe final archived official accuracy; independent prefixes provisionally retained",
    "features": "exact whitelisted operational scalar fields plus ordered typed declared closure",
    "not_claimed": [
        "full-recipe semantic approval",
        "executed-artifact fidelity",
        "RPM gain",
        "heldout performance",
        "prospective simultaneous recipe choice",
    ],
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _value_end(text, start):
    """Lexically skip a JSON value without decoding its potentially private value."""
    depth, quoted, escape = 0, False, False
    for i in range(start, len(text)):
        char = text[i]
        if quoted:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            if depth == 0:
                return i
            depth -= 1
        elif char == "," and depth == 0:
            return i
    return len(text)


def _cell_from_json_line(text):
    """Decode only top-level keys and cell_id; labels are never decoded here."""
    decoder = json.JSONDecoder()
    cursor = 1
    if not text.lstrip().startswith("{"):
        raise ValueError("Inventory line must be a JSON object")
    cursor = text.index("{") + 1
    while cursor < len(text):
        while cursor < len(text) and text[cursor] in " \r\n\t,":
            cursor += 1
        if text[cursor] == "}":
            break
        key, cursor = decoder.raw_decode(text, cursor)
        while text[cursor].isspace():
            cursor += 1
        if text[cursor] != ":":
            raise ValueError("Malformed top-level inventory field")
        cursor += 1
        while text[cursor].isspace():
            cursor += 1
        if key == "cell_id":
            cell, _ = decoder.raw_decode(text, cursor)
            if not isinstance(cell, str):
                raise ValueError("cell_id must be a string")
            return cell
        cursor = _value_end(text, cursor)
    raise ValueError("Missing top-level cell_id")


def load_train_inventory(path, split):
    """TEST lines are skipped before json.loads can decode their labels/features."""
    partition = split["cell_partition"]
    result, skipped = [], Counter()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            cell = _cell_from_json_line(line)
            if cell not in partition:
                raise ValueError("Inventory contains a run outside the frozen split")
            if partition[cell] != "train":
                skipped[cell] += 1
                continue
            row = json.loads(line)
            if row["cell_id"] != cell:
                raise ValueError("Ambiguous duplicate top-level cell_id")
            result.append(row)
    return result, dict(skipped)


def _scalar(value):
    if isinstance(value, str):
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.strip()):
            return None
        value = float(value)
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def parameter_features(row):
    """Positive access only; free-form other/precision explanations are ignored."""
    task = row["model_input"]["task"]
    base = task.get("base_model")
    role = row.get("role")
    if base not in BASES or role not in ROLES:
        raise ValueError("Unknown declared base or step role")
    setup = row["model_input"]["plan"].get("setup", {})
    method = setup.get("method", {})
    hp = method.get("hyperparams", {}) if isinstance(method, dict) else {}
    hp = hp if isinstance(hp, dict) else {}
    result = {"base": base, "role": role}
    for key in sorted(NUMERIC_KEYS):
        value = _scalar(hp.get(key))
        if value is not None:
            result[key] = value
            result[key + "_declared"] = 1.0
    for key, allowed in CATEGORIES.items():
        value = hp.get(key)
        if isinstance(value, str) and value.strip().lower() in allowed:
            result[key] = allowed[value.strip().lower()]
    return result


def _data_dependency_screen(row):
    """Source-category screening only, never a feature or a numeric-yield reader."""
    setup = row["model_input"]["plan"].get("setup", {})
    data = setup.get("data", [])
    if not isinstance(data, list) or (row.get("role") == "training" and not data):
        return ["missing_declared_external_data_sources"]
    reasons = []
    for entry in data:
        source = entry.get("source") if isinstance(entry, dict) else None
        if not isinstance(source, str):
            reasons.append("unknown_declared_data_source_category")
            continue
        if SELF_DATA.search(source):
            reasons.append("internal_self_generated_data_operation_unmodeled")
        elif not any(name in source.lower() for name in EXTERNAL_DATASETS) and (
            row.get("role") == "training" or not source.lower().startswith("none")
        ):
            reasons.append("unknown_declared_data_source_category")
        command = entry.get("build_command", [])
        tokens = command if isinstance(command, list) else re.split(r"\s+", str(command))
        if any(
            isinstance(token, str)
            and (
                token.split("=", 1)[0].replace("_", "-")
                in {
                    "--rft",
                    "--model",
                    "--model-path",
                    "--checkpoint",
                    "--ckpt",
                    "--init",
                    "--init-from",
                }
                or token.startswith(("--rft_", "--rft-"))
            )
            for token in tokens
        ):
            reasons.append("data_builder_model_or_rft_dependency_unmodeled")
    return reasons


def declared_closure(target_id, rows, graph):
    """Validate a complete declared weight/config DAG, not execution identity."""
    nodes = graph["nodes"]
    target = rows[target_id]
    pending, visited, derived, reasons = set(), set(), set(), []

    def visit(identity):
        if identity in pending:
            reasons.append("declared_dependency_cycle")
            return
        if identity in visited:
            return
        if identity not in rows or identity not in nodes:
            reasons.append("closure_step_missing_from_train_inventory")
            return
        pending.add(identity)
        row, node = rows[identity], nodes[identity]
        if row["cell_id"] != target["cell_id"] or node["cell_id"] != target["cell_id"]:
            reasons.append("cross_run_dependency")
        if row["audit"].get("eligible") is not True:
            reasons.append("closure_plan_ineligible:" + identity)
        reasons.extend(_data_dependency_screen(row))
        if row.get("role") in {"training", "merge"} and not any(
            edge.get("kind") == "weights" for edge in node["parents"]
        ):
            reasons.append("missing_declared_weight_origin")
        if node.get("closure_has_unresolved_dependencies"):
            reasons.append("unresolved_declared_closure")
        if node.get("operation_expansion_required"):
            reasons.append("operation_expansion_required")
        try:
            parameter_features(row)
        except ValueError:
            reasons.append("unknown_base_or_role:" + identity)
        if node.get("base_model") != row["model_input"]["task"].get("base_model"):
            reasons.append("graph_task_base_mismatch")
        for edge in node["parents"]:
            scope, status = edge.get("dependency_scope"), edge.get("resolution_status")
            producer = edge.get("producer_id")
            if edge.get("card_only"):
                reasons.append("card_only_dependency")
            if scope == "data_builder_only":
                reasons.append("data_builder_only_requires_safe_expansion")
            if status == "declared_base_model":
                if producer or scope not in {"requires_producer_weights", "configuration_only"}:
                    reasons.append("invalid_base_edge")
            elif status == "internal_declared_operation":
                if producer or scope != "in_recipe_operation":
                    reasons.append("invalid_internal_edge")
            elif status == "time_qualified_declared_artifact":
                if edge.get("kind") not in {"weights", "configuration"}:
                    reasons.append("external_data_operation_not_modeled")
                if scope not in {"requires_producer_weights", "configuration_only"}:
                    reasons.append("unsupported_dependency_scope")
                if edge.get("artifact_variant") not in {"final", "declared_output_root"}:
                    reasons.append("artifact_variant_training_prefix_unresolved")
                if not producer:
                    reasons.append("missing_declared_producer")
                else:
                    visit(producer)
            else:
                reasons.append("unresolved_edge")
        pending.remove(identity)
        visited.add(identity)
        derived.add(identity)

    if target_id not in nodes:
        return [], ["target_missing_from_graph"]
    visit(target_id)
    declared = nodes[target_id].get("topological_closure", [])
    if (
        len(declared) != len(set(declared))
        or set(declared) != derived
        or not declared
        or declared[-1] != target_id
    ):
        reasons.append("incomplete_or_ambiguous_declared_closure")
    positions = {identity: i for i, identity in enumerate(declared)}
    for identity in derived:
        for edge in nodes[identity]["parents"]:
            producer = edge.get("producer_id")
            if (
                producer
                and producer in derived
                and positions.get(producer, -1) >= positions.get(identity, -1)
            ):
                reasons.append("invalid_declared_topological_order")
    return declared, sorted(set(reasons))


def recipe_features(closure, rows, graph):
    result = {"closure_steps": float(len(closure))}
    counts = Counter()
    for i, identity in enumerate(closure):
        for key, value in parameter_features(rows[identity]).items():
            result[f"step_{i}.{key}"] = value
        edges = {
            (
                e.get("kind"),
                e.get("dependency_scope"),
                e.get("producer_id"),
                e.get("artifact_variant"),
            )
            for e in graph["nodes"][identity]["parents"]
        }
        for kind, scope, _, _ in edges:
            if kind in {"weights", "configuration", "generated_data"} and scope in {
                "in_recipe_operation",
                "requires_producer_weights",
                "configuration_only",
            }:
                counts[f"edge_count.{kind}.{scope}"] += 1
    result.update({key: float(value) for key, value in counts.items()})
    return result


def prepare_cohort(rows, split, graph):
    """Filter TEST first, then screen declarations, then read final target labels."""
    partition = split["cell_partition"]
    train = {}
    for row in rows:
        cell = row["cell_id"]
        if cell not in partition:
            raise ValueError("Unknown frozen run")
        if partition[cell] != "train":
            continue
        identity = row["example_id"]
        if identity in train:
            raise ValueError("Duplicate TRAIN example ID")
        train[identity] = row
    cohort, omitted = [], []
    for identity, row in sorted(train.items()):
        closure, reasons = declared_closure(identity, train, graph)
        if row["audit"].get("eligible") is not True:
            reasons.append("target_ineligible")
        if reasons:
            omitted.append(
                {"example_id": identity, "cell_id": row["cell_id"], "reasons": sorted(set(reasons))}
            )
            continue
        # The only label read in the feature/cohort path, after both filters.
        value = (row.get("label") or {}).get("accuracy")
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            omitted.append(
                {
                    "example_id": identity,
                    "cell_id": row["cell_id"],
                    "reasons": ["missing_or_invalid_final_target_label"],
                }
            )
            continue
        cohort.append(
            {
                "example_id": identity,
                "cell_id": row["cell_id"],
                "benchmark": row["model_input"]["task"]["benchmark"],
                "closure_ids_private": closure,
                "features": recipe_features(closure, train, graph),
                "last_step_features": parameter_features(row),
                "target": float(value),
                "fidelity_status": "provisional_declared_recipe_not_executed_artifact_verified",
            }
        )
    return cohort, omitted


def _folds(cells, benchmark):
    ordered = sorted(cells, key=lambda cell: digest([SEED, benchmark, cell]))
    return {cell: i % 4 for i, cell in enumerate(ordered)}


def _weights(items):
    counts = Counter(item["cell_id"] for item in items)
    # Normalize to mean one; otherwise Ridge alpha would vary with run count.
    raw = np.array([1 / counts[item["cell_id"]] for item in items])
    return raw / raw.mean()


def _model_predict(train, valid, variant):
    y = np.array([item["target"] for item in train])
    weights = _weights(train)
    if variant == "equal_run_mean":
        return np.full(len(valid), np.average(y, weights=weights)), {"feature_names": []}
    field = "last_step_features" if variant == "extra_trees_last_step" else "features"
    vectorizer = DictVectorizer(sparse=True, sort=True)
    x = vectorizer.fit_transform([item[field] for item in train])
    xv = vectorizer.transform([item[field] for item in valid])
    if variant == "ridge_full":
        scaler = StandardScaler(with_mean=False)
        x = scaler.fit_transform(x, sample_weight=weights)
        xv = scaler.transform(xv)
        model = Ridge(**SPEC["ridge"])
    else:
        model = ExtraTreesRegressor(**SPEC["extra_trees"], random_state=SEED)
    model.fit(x, y, sample_weight=weights)
    predictions = np.clip(model.predict(xv), 0, 1)
    return predictions, {"feature_names": list(vectorizer.get_feature_names_out())}


def _metrics(rows, variant):
    runs = {}
    for row in rows:
        runs.setdefault(row["cell_id"], []).append(row)
    errors = [
        np.mean([abs(r["target"] - r["predictions"][variant]) for r in values])
        for values in runs.values()
    ]
    choices = []
    for cell, values in sorted(runs.items()):
        if len(values) < 2:
            continue
        chosen = min(
            values, key=lambda r: (-r["predictions"][variant], digest([SEED, r["example_id"]]))
        )
        oracle = max(r["target"] for r in values)
        choices.append(
            {
                "cell_id": cell,
                "candidate_count": len(values),
                "selected_example_id": chosen["example_id"],
                "selected_final_accuracy": chosen["target"],
                "oracle_final_accuracy": oracle,
                "regret": oracle - chosen["target"],
                "uniform_candidate_expected_accuracy": float(
                    np.mean([r["target"] for r in values])
                ),
            }
        )
    return {
        "runs": len(runs),
        "recipes": len(rows),
        "equal_run_mae": float(np.mean(errors)),
        "selection_runs_with_two_or_more_candidates": len(choices),
        "selected_final_accuracy": float(np.mean([c["selected_final_accuracy"] for c in choices]))
        if choices
        else None,
        "selection_regret": float(np.mean([c["regret"] for c in choices])) if choices else None,
        "selection_by_run_private": choices,
    }


def run_pilot(rows, split, graph):
    """Fit only TRAIN-run CV models; return private JSON-serializable results."""
    cohort, omitted = prepare_cohort(rows, split, graph)
    variants = ["equal_run_mean", "ridge_full", "extra_trees_full", "extra_trees_last_step"]
    predictions, folds, metrics = [], [], {}
    for benchmark in sorted({r["benchmark"] for r in cohort}):
        items = [r for r in cohort if r["benchmark"] == benchmark]
        cells = {r["cell_id"] for r in items}
        if len(cells) < 4:
            raise ValueError(
                "Four-fold TRAIN-run CV requires at least four retained runs per benchmark"
            )
        assignment = _folds(cells, benchmark)
        benchmark_predictions = []
        for fold in range(4):
            train = [r for r in items if assignment[r["cell_id"]] != fold]
            valid = [r for r in items if assignment[r["cell_id"]] == fold]
            outputs, vocabularies = {}, {}
            for variant in variants:
                outputs[variant], vocabularies[variant] = _model_predict(train, valid, variant)
            folds.append(
                {
                    "benchmark": benchmark,
                    "fold": fold,
                    "train_cells": sorted({r["cell_id"] for r in train}),
                    "validation_cells": sorted({r["cell_id"] for r in valid}),
                    "train_examples": [r["example_id"] for r in train],
                    "validation_examples": [r["example_id"] for r in valid],
                    "fold_train_feature_vocabularies": vocabularies,
                }
            )
            for i, item in enumerate(valid):
                benchmark_predictions.append(
                    {
                        "example_id": item["example_id"],
                        "cell_id": item["cell_id"],
                        "benchmark": benchmark,
                        "fold": fold,
                        "target": item["target"],
                        "predictions": {v: float(outputs[v][i]) for v in variants},
                    }
                )
        predictions.extend(benchmark_predictions)
        metrics[benchmark] = {v: _metrics(benchmark_predictions, v) for v in variants}
    declared_train = {cell for cell, part in split["cell_partition"].items() if part == "train"}
    retained = {row["cell_id"] for row in cohort}
    return {
        "schema": "wm-parameter-pilot-v1",
        "spec": copy.deepcopy(SPEC),
        "metrics": metrics,
        "cohort_private": cohort,
        "omitted_train_cards": omitted,
        "train_runs_without_retained_targets": sorted(declared_train - retained),
        "train_runs_with_retained_targets": sorted(retained),
        "folds_private": folds,
        "predictions_private": sorted(predictions, key=lambda r: r["example_id"]),
        "cohort_feature_sha256": digest(
            [{k: r[k] for k in ("example_id", "features", "last_step_features")} for r in cohort]
        ),
        "cohort_target_sha256_private": digest(
            [{k: r[k] for k in ("example_id", "target")} for r in cohort]
        ),
        "test_scored": False,
        "heldout_labels_accessed": False,
        "model_saved_or_served": False,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "limits": [
            "Limited parameter features omit data recipes, code, and operational prose.",
            "Cohort excludes ambiguous/variant/data-operation dependency closures.",
            "Declared lineage is not execution or archived-label identity verification.",
            "Within-run choices are retrospective adaptive prefixes, not simultaneous candidate batches.",
            "No confidence interval or population-wide/full-recipe/RPM claim is made.",
        ],
    }
