"""CPU-only signed-delta pilot, with frozen whole-run sibling evaluation.

This is a limited-feature, retrospective known-parent experiment. Official parent
grades are GIVEN, not claimed historically available when recipes were proposed.
No private narrative or arbitrary code text is a feature. Exact declared parent
paths are matched to recorded final output paths; checkpoint bytes are not certified.
Old wm_study and final-target-only artifacts are never changed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from tools.outcome_prediction.wm_grade_inventory import valid_official_label

SEED = 20260905
NAMES = (
    "carry_parent",
    "ridge_current_delta",
    "ridge_history_delta",
    "trees_history_delta",
    "ridge_history_absolute",
)
HP = (
    "lr",
    "learning_rate",
    "epochs",
    "num_train_epochs",
    "batch_size",
    "per_device_train_batch_size",
    "grad_accum",
    "gradient_accumulation_steps",
    "max_seq_len",
    "max_seq_length",
    "max_steps",
    "warmup",
    "warmup_ratio",
    "weight_decay",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "beta",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "repetition_penalty",
)
FAMILIES = ("sft", "rft", "grpo", "ppo", "dpo", "distill", "decoding", "evaluation", "merge")
DATASETS = (
    "gsm8k",
    "metamath",
    "openmathinstruct",
    "openmathreasoning",
    "numina",
    "orca-math",
    "openr1",
    "mixture-of-thoughts",
)
CODE_NAMES = (
    "SFTTrainer",
    "GRPOTrainer",
    "DPOTrainer",
    "LoraConfig",
    "get_peft_model",
    "train_on_responses_only",
    "apply_chat_template",
    "DataCollatorForCompletionOnlyLM",
    "DataCollatorForLanguageModeling",
    "load_dataset",
    "generate",
    "set_seed",
)
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Naive observation timestamp")
    return result


def number(value):
    if type(value) in (int, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str) and NUMBER.fullmatch(value.strip()):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def positive_features(row):
    """Read only whitelisted first-plan settings and syntax marker names.

    n_examples, realized progress, prose, code constants, IDs and all outcomes are
    intentionally absent. Numeric hyperparameters are not blanket-redacted.
    """
    source = row["model_input"]
    setup = source["plan"].get("setup", {})
    method = setup.get("method") or {}
    hp = method.get("hyperparams") or {}
    family = str(method.get("family") or "").lower().replace("-", "_")
    family = family if family in FAMILIES else "other"
    result = {"family." + family: 1.0}
    for key in HP:
        value = number(hp.get(key))
        if value is not None:
            result["hp." + key] = value
    for key, allowed in {
        "precision": ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"),
        "scheduler": ("cosine", "linear", "constant", "constant_with_warmup"),
    }.items():
        value = hp.get(key)
        if isinstance(value, str) and value.strip().lower() in allowed:
            result[key + "." + value.strip().lower()] = 1.0
    data = setup.get("data") or []
    for category in DATASETS:
        result["dataset." + category] = float(
            any(category in str(d.get("source") or "").lower() for d in data if isinstance(d, dict))
        )
    result["n_declared_data_entries"] = float(len(data))
    code_names, parsed, missing = set(), 0, 0
    for script in source.get("code", []):
        if script.get("status") != "reconstructed" or not isinstance(script.get("content"), str):
            missing += 1
            continue
        try:
            tree = ast.parse(script["content"])
        except (SyntaxError, ValueError):
            missing += 1
            continue
        parsed += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                code_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                code_names.add(node.attr)
    result["code.parsed_scripts"] = float(parsed)
    result["code.unavailable_or_nonpython"] = float(missing)
    for name in CODE_NAMES:
        result["code.syntax." + name] = float(name in code_names)
    return result


def parent_edge(node):
    """Student weight input only, never a data generator or same-card alias."""
    if node.get("is_merge_declared"):
        raise ValueError("multiple_parent_merge")
    edges = [e for e in node["parents"] if e["kind"] == "weights" and not e.get("internal")]
    if len(edges) != 1:
        raise ValueError("not_exactly_one_weight_parent")
    edge = edges[0]
    if edge.get("resolution_status") == "declared_base_model":
        raise ValueError("base_parent_score_not_recorded")
    if (
        edge.get("resolution_status") != "time_qualified_declared_artifact"
        or edge.get("card_only")
        or not edge.get("artifact")
        or not edge.get("producer_id")
    ):
        raise ValueError("weight_parent_unresolved")
    return edge


def official_score(row):
    label = row.get("label")
    if not isinstance(label, dict):
        raise TypeError("missing_official_grade")
    value = number(label.get("accuracy"))
    if (
        not valid_official_label(label)
        or type(label.get("evaluation_n")) is not int
        or label.get("evaluation_n") != row["task"]["evaluation_n"]
    ):
        raise ValueError("invalid_official_grade")
    return value


def closed_times(rows):
    """Ledger event times only: no final-card outcome parsing."""
    times = {}
    paths = sorted({r["provenance"]["ledger_path"] for r in rows.values()})
    for path in paths:
        cell = Path(path).parent.parent.name
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if (
                event.get("event") == "submit"
                and event.get("stage") == "closed"
                and event.get("at")
            ):
                key = cell + "/" + event["card_id"]
                at = timestamp(event["at"])
                times[key] = min(times.get(key, at), at)
    return times, {p: sha(p) for p in paths}


def history_relations(parent_id, nodes):
    """Keep data/configuration ancestry distinct from student weight ancestry."""
    roles = defaultdict(set)
    pending = [(parent_id, "parent")]
    visited = set()
    while pending:
        node_id, role = pending.pop()
        if (node_id, role) in visited:
            continue
        visited.add((node_id, role))
        roles[node_id].add(role)
        for edge in nodes[node_id]["parents"]:
            producer = edge.get("producer_id")
            if producer is not None and producer in nodes and producer != node_id:
                next_role = edge["kind"] if role in ("parent", "weights") else role
                pending.append((producer, next_role))
    return {key: "+".join(sorted(value)) for key, value in roles.items()}


def build_inputs(rows, nodes, split, availability, closed):
    """Cohort uses grade availability, never child-score magnitudes.

    History is the resolved dependency closure of the common parent, excluding
    candidate-specific later observations. Unknown dependencies remain explicit.
    """
    examples, excluded = [], []
    for example_id, row in sorted(rows.items()):
        try:
            if not row["eligible"]:
                raise ValueError("initial_fidelity_screen_rejected")
            if not availability.get(example_id, False):
                raise ValueError("child_official_grade_unavailable")
            node = nodes[example_id]
            edge = parent_edge(node)
            parent_id = edge["producer_id"]
            parent = rows[parent_id]
            if parent["cell_id"] != row["cell_id"]:
                raise ValueError("cross_run_parent")
            if not availability.get(parent_id, False):
                raise ValueError("parent_official_grade_unavailable")
            if parent["task"] != row["task"]:
                raise ValueError("parent_metric_context_mismatch")
            output = parent["audit"]["output_artifact_comparison"]["final_output_checkpoint"]
            if not isinstance(output, str) or edge["artifact"].rstrip("/") != output.rstrip("/"):
                raise ValueError("parent_score_artifact_path_mismatch")
            cutoff = timestamp(row["first_submitted_at"])
            if parent_id not in closed or closed[parent_id] >= cutoff:
                raise ValueError("parent_not_closed_before_proposal")
            parent_node = nodes[parent_id]
            history_ids = sorted(
                parent_node["topological_closure"],
                key=lambda key: (timestamp(rows[key]["first_submitted_at"]), key),
            )
            if (
                example_id in history_ids
                or parent_id not in history_ids
                or len(set(history_ids)) != len(history_ids)
            ):
                raise ValueError("invalid_parent_history")
            parent_cutoff = closed[parent_id]
            relations = history_relations(parent_id, nodes)
            history = []
            for history_id in history_ids:
                hr = rows[history_id]
                if hr["cell_id"] != row["cell_id"] or timestamp(hr["first_submitted_at"]) >= cutoff:
                    raise ValueError("noncausal_parent_history")
                # Fixed at parent completion, so both siblings see byte-identical state.
                observed = (
                    availability.get(history_id, False)
                    and history_id in closed
                    and closed[history_id] <= parent_cutoff
                )
                role = relations.get(history_id, "unresolved_ancestor")
                features = positive_features(hr)
                features["graph.unresolved_dependencies"] = float(
                    nodes[history_id]["closure_has_unresolved_dependencies"]
                )
                history.append(
                    {
                        "features": features,
                        "accuracy": official_score(hr) if observed else None,
                        "relation": role,
                    }
                )
            current = positive_features(row)
            current["graph.unresolved_dependencies"] = float(
                node["closure_has_unresolved_dependencies"]
            )
            parent_key = digest(
                {
                    "cell": row["cell_id"],
                    "task": row["task"],
                    "producer": parent_id,
                    "artifact": edge["artifact"].rstrip("/"),
                }
            )
            history_hash = digest(history)
            examples.append(
                {
                    "example_id": example_id,
                    "cell_id": row["cell_id"],
                    "benchmark": row["benchmark"],
                    "partition": split["cell_partition"][row["cell_id"]],
                    "parent_key": parent_key,
                    "parent_id": parent_id,
                    "parent_artifact": edge["artifact"],
                    "parent_accuracy": official_score(parent),
                    "current_features": current,
                    "history": history,
                    "history_sha256": history_hash,
                    "history_ids": history_ids,
                    "history_fidelity": {
                        "initial_screen_rejected": [
                            key for key in history_ids if not rows[key]["eligible"]
                        ],
                        "first_final_setup_changed": [
                            key
                            for key in history_ids
                            if rows[key]["audit"].get("first_final_setup_changed", False)
                        ],
                    },
                    "candidate_closure": node["topological_closure"],
                    "strict_declared_closure": not node["closure_has_unresolved_dependencies"],
                    "first_submitted_at": row["first_submitted_at"],
                    "parent_closed_at": parent_cutoff.isoformat(),
                }
            )
        except (ValueError, TypeError, KeyError) as error:
            excluded.append(
                {
                    "example_id": example_id,
                    "cell_id": row["cell_id"],
                    "partition": split["cell_partition"][row["cell_id"]],
                    "reason": str(error),
                }
            )
    return examples, excluded


def sibling_pairs(examples):
    groups = defaultdict(list)
    for row in examples:
        groups[(row["partition"], row["benchmark"], row["cell_id"], row["parent_key"])].append(row)
    pairs, excluded = [], []
    for group in groups.values():
        for a, b in itertools.combinations(sorted(group, key=lambda r: r["example_id"]), 2):
            reason = None
            if (
                a["history_sha256"] != b["history_sha256"]
                or a["parent_accuracy"] != b["parent_accuracy"]
            ):
                reason = "different_parent_observation_history"
            elif (
                a["example_id"] in b["candidate_closure"]
                or b["example_id"] in a["candidate_closure"]
            ):
                reason = "one_candidate_is_dependency_of_other"
            if reason:
                excluded.append({"a": a["example_id"], "b": b["example_id"], "reason": reason})
                continue
            pairs.append(
                {
                    "pair_id": digest([a["example_id"], b["example_id"], a["parent_key"]]),
                    "a": a["example_id"],
                    "b": b["example_id"],
                    "cell_id": a["cell_id"],
                    "benchmark": a["benchmark"],
                    "partition": a["partition"],
                    "parent_key": a["parent_key"],
                    "history_sha256": a["history_sha256"],
                    "strict_declared_closure": a["strict_declared_closure"]
                    and b["strict_declared_closure"],
                }
            )
    return pairs, excluded


def prepare(policy_path, output):
    policy = read(policy_path)
    for name, source in policy["sources"].items():
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Changed frozen source: " + name)
    source = {k: v["path"] for k, v in policy["sources"].items()}
    rows = {
        r["example_id"]: r
        for r in (
            json.loads(line) for line in Path(source["inventory"]).read_text().splitlines() if line
        )
    }
    nodes, split = read(source["graph"])["nodes"], read(source["split"])
    availability = {
        r["example_id"]: r["has_valid_official_final_grade"]
        for r in read(source["availability"])["records"]
    }
    closed, ledger_hashes = closed_times(rows)
    examples, excluded = build_inputs(rows, nodes, split, availability, closed)
    pairs, pair_exclusions = sibling_pairs(examples)
    pair_members = {p[key] for p in pairs if p["partition"] == "test" for key in ("a", "b")}
    unpaired = [
        r for r in examples if r["partition"] == "test" and r["example_id"] not in pair_members
    ]
    for row in unpaired:
        excluded.append(
            {
                "example_id": row["example_id"],
                "cell_id": row["cell_id"],
                "partition": "test",
                "reason": "no_eligible_heldout_sibling_pair",
            }
        )
    examples = [r for r in examples if r["partition"] == "train" or r["example_id"] in pair_members]
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_new(output / "policy.json", policy)
    write_new(output / "inputs.json", examples)
    write_new(output / "pairs.json", pairs)
    train_labels = {
        r["example_id"]: official_score(rows[r["example_id"]])
        for r in examples
        if r["partition"] == "train"
    }
    write_new(output / "train_labels.json", train_labels)
    coverage = {
        "examples": len(examples),
        "pairs": len(pairs),
        "excluded": excluded,
        "pair_exclusions": pair_exclusions,
        "run_partitions": split["cell_partition"],
        "by_partition_benchmark": {
            p + "/" + b: {
                "examples": sum(r["partition"] == p and r["benchmark"] == b for r in examples),
                "runs": len(
                    {r["cell_id"] for r in examples if r["partition"] == p and r["benchmark"] == b}
                ),
                "pairs": sum(r["partition"] == p and r["benchmark"] == b for r in pairs),
            }
            for p in ("train", "test")
            for b in ("gsm8k", "aime2025")
        },
    }
    write_new(output / "coverage.json", coverage)
    write_new(
        output / "manifest.json",
        {
            "policy_source_sha256": sha(policy_path),
            "ledger_sha256": ledger_hashes,
            "files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "child_score_magnitudes_used_for_cohort": False,
            "test_target_labels_exported": False,
            "private_inventory_objects_decoded": True,
        },
    )
    return coverage["by_partition_benchmark"]


def metrics(examples, pairs, labels, predictions):
    by_id = {r["example_id"]: r for r in examples}
    errors = defaultdict(list)
    for r in examples:
        truth = labels[r["example_id"]] - r["parent_accuracy"]
        errors[r["cell_id"]].append(abs(predictions[r["example_id"]] - truth))
    details = []
    for p in pairs:
        a, b = p["a"], p["b"]
        ya, yb = labels[a], labels[b]
        pa, pb = predictions[a], predictions[b]
        true_tie, predicted_tie = abs(ya - yb) <= 1e-12, abs(pa - pb) <= 1e-12
        chosen = (ya + yb) / 2 if predicted_tie else (ya if pa > pb else yb)
        details.append(
            {
                **p,
                "correct": None
                if true_tie
                else (0.5 if predicted_tie else float((pa > pb) == (ya > yb))),
                "true_tie": true_tie,
                "predicted_tie": predicted_tie,
                "chosen_accuracy": chosen,
                "regret": max(ya, yb) - chosen,
                "predicted_delta_a": pa,
                "predicted_delta_b": pb,
                "true_delta_a": ya - by_id[a]["parent_accuracy"],
                "true_delta_b": yb - by_id[b]["parent_accuracy"],
            }
        )

    def macro(key):
        groups = defaultdict(list)
        for d in details:
            if d[key] is not None:
                groups[d["cell_id"]].append(d[key])
        return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None

    return {
        "n_examples": len(examples),
        "n_pairs": len(pairs),
        "n_pair_runs": len({p["cell_id"] for p in pairs}),
        "true_ties": sum(d["true_tie"] for d in details),
        "predicted_ties": sum(d["predicted_tie"] for d in details),
        "delta_mae": float(np.mean([np.mean(v) for v in errors.values()])) if errors else None,
        "pair_accuracy": macro("correct"),
        "chosen_accuracy": macro("chosen_accuracy"),
        "regret": macro("regret"),
        "pairs": details,
    }


def paired_intervals(reference, treatment, repeats=2000):
    """Equal-run paired bootstrap; descriptive only, especially with two runs."""
    a = {p["pair_id"]: p for p in reference["pairs"]}
    groups = defaultdict(list)
    for p in treatment["pairs"]:
        r = a[p["pair_id"]]
        groups[p["cell_id"]].append(p["chosen_accuracy"] - r["chosen_accuracy"])
    if not groups:
        return None
    changes = np.array([np.mean(groups[k]) for k in sorted(groups)])
    rng = np.random.default_rng(SEED)
    samples = changes[rng.integers(0, len(changes), size=(repeats, len(changes)))].mean(axis=1)
    return {
        "n_runs": len(changes),
        "gain": float(changes.mean()),
        "ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
        "replicates": repeats,
        "warning": "descriptive; few independent runs; not adjusted for model comparisons",
    }


def fit_and_evaluate(bundle, output):
    from tools.outcome_prediction.wm_delta_model import LightweightDeltaModel

    bundle, output = Path(bundle), Path(output)
    manifest = read(bundle / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        if sha(bundle / name) != expected:
            raise ValueError("Changed prepared input: " + name)
    policy = read(bundle / "policy.json")
    for name, source in policy["sources"].items():
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Changed source before fit: " + name)
    examples, pairs, train_labels = (
        read(bundle / "inputs.json"),
        read(bundle / "pairs.json"),
        read(bundle / "train_labels.json"),
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    predictions, training = {}, {}
    for benchmark in ("gsm8k", "aime2025"):
        tr = [r for r in examples if r["partition"] == "train" and r["benchmark"] == benchmark]
        te = [r for r in examples if r["partition"] == "test" and r["benchmark"] == benchmark]
        assert not {r["cell_id"] for r in tr} & {r["cell_id"] for r in te}
        if not tr or not te:
            continue
        targets = [train_labels[r["example_id"]] for r in tr]
        predictions[benchmark], training[benchmark] = {}, {}
        for name in NAMES:
            start = time.perf_counter()
            model = LightweightDeltaModel(name).fit(tr, targets)
            pred = model.predict_delta(te)
            model_path = output / (benchmark + "__" + name + ".joblib")
            joblib.dump(model, model_path)
            model_path.chmod(0o600)
            predictions[benchmark][name] = {r["example_id"]: float(v) for r, v in zip(te, pred)}
            training[benchmark][name] = {
                "metadata": model.metadata(),
                "seconds_fit_predict_save": time.perf_counter() - start,
                "serialized_bytes": model_path.stat().st_size,
                "model_sha256": sha(model_path),
                "train_ids": [r["example_id"] for r in tr],
            }
    # Freeze all model choices and predictions BEFORE reading TEST targets for scoring.
    write_new(output / "predictions.json", predictions)
    write_new(output / "training.json", training)
    write_new(
        output / "prediction_freeze.json",
        {
            "policy_sha256": sha(bundle / "policy.json"),
            "inputs_sha256": sha(bundle / "inputs.json"),
            "pairs_sha256": sha(bundle / "pairs.json"),
            "predictions_sha256": sha(output / "predictions.json"),
            "training_sha256": sha(output / "training.json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hyperparameter_or_test_selection": False,
        },
    )
    needed = {r["example_id"] for r in examples if r["partition"] == "test"}
    labels = {}
    inventory_raw = Path(policy["sources"]["inventory"]["path"]).read_bytes()
    if hashlib.sha256(inventory_raw).hexdigest() != policy["sources"]["inventory"]["sha256"]:
        raise ValueError("Inventory changed between fitting and held-out scoring")
    for line in inventory_raw.decode().splitlines():
        row = json.loads(line)
        if row["example_id"] in needed:
            labels[row["example_id"]] = official_score(row)
    result = {}
    for benchmark, arms in predictions.items():
        te = [r for r in examples if r["partition"] == "test" and r["benchmark"] == benchmark]
        ps = [p for p in pairs if p["partition"] == "test" and p["benchmark"] == benchmark]
        result[benchmark] = {name: metrics(te, ps, labels, pred) for name, pred in arms.items()}
        for name, values in result[benchmark].items():
            values["selected_accuracy_vs_carry"] = paired_intervals(
                result[benchmark]["carry_parent"], values
            )
            strict_pairs = [p for p in ps if p["strict_declared_closure"]]
            strict_ids = {p[k] for p in strict_pairs for k in ("a", "b")}
            values["strict_declared_closure_subset"] = metrics(
                [r for r in te if r["example_id"] in strict_ids], strict_pairs, labels, arms[name]
            )
    write_new(output / "report.json", result)
    return {
        b: {n: {k: v for k, v in m.items() if k != "pairs"} for n, m in arms.items()}
        for b, arms in result.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--policy", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = subs.add_parser("run")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        prepare(args.policy, args.output)
        if args.command == "prepare"
        else fit_and_evaluate(args.bundle, args.output)
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
