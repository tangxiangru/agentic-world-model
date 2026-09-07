"""Compare frozen and learned RPM-style judges on identical held-out pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .rpm_judge import digest, read_rows, write_json
except ImportError:
    from rpm_judge import digest, read_rows, write_json


def per_cell(records, method):
    grouped = defaultdict(list)
    for row in records:
        p = row["probabilities"][method]
        label = row["y_a"] > row["y_b"]
        forced = row.get("choices", {}).get(method)
        credit = (
            float(forced == label)
            if forced is not None
            else 0.5
            if abs(p - 0.5) < 1e-12
            else float((p > 0.5) == label)
        )
        q = np.clip(p, 1e-8, 1 - 1e-8)
        grouped[row["cell_id"]].append(
            [credit, (1 - credit) * row["gap"], (p - label) ** 2, -np.log(q if label else 1 - q)]
        )
    return {
        cell: {"n": len(values), "mean": np.mean(values, axis=0).tolist()}
        for cell, values in sorted(grouped.items())
    }


def summarize(records, method, bootstraps=5000):
    cells = per_cell(records, method)
    if not cells:
        return {"pairs": 0, "cells": 0}
    values = np.array([r["mean"] for r in cells.values()])
    weights = np.array([r["n"] for r in cells.values()])
    draws = np.random.default_rng(20260905).integers(0, len(cells), (bootstraps, len(cells)))
    out = {"pairs": len(records), "cells": len(cells), "per_cell": cells}
    for index, name in enumerate(("accuracy", "regret", "brier", "log_loss")):
        macro = values[:, index]
        out["macro_" + name] = float(macro.mean())
        out["micro_" + name] = float(np.average(macro, weights=weights))
        out["macro_" + name + "_ci95"] = np.quantile(
            macro[draws].mean(axis=1), [0.025, 0.975]
        ).tolist()
        micro_draws = (macro[draws] * weights[draws]).sum(axis=1) / weights[draws].sum(axis=1)
        out["micro_" + name + "_ci95"] = np.quantile(micro_draws, [0.025, 0.975]).tolist()
    return out


def paired_difference(records, method, reference, bootstraps=5000):
    first, second = per_cell(records, method), per_cell(records, reference)
    cells = sorted(first.keys() & second.keys())
    if not cells:
        return {"pairs": 0, "cells": 0}
    a = np.array([first[c]["mean"] for c in cells])
    b = np.array([second[c]["mean"] for c in cells])
    weights = np.array([first[c]["n"] for c in cells])
    # Positive always means improvement: accuracy increases, losses decrease.
    delta = (a - b) * np.array([1, -1, -1, -1])
    draws = np.random.default_rng(20260905).integers(0, len(cells), (bootstraps, len(cells)))
    out = {"pairs": len(records), "cells": len(cells), "method": method, "reference": reference}
    for i, name in enumerate(
        ("accuracy_gain", "regret_reduction", "brier_reduction", "log_loss_reduction")
    ):
        macro = delta[:, i]
        out["macro_" + name] = float(macro.mean())
        out["macro_" + name + "_ci95"] = np.quantile(
            macro[draws].mean(axis=1), [0.025, 0.975]
        ).tolist()
        out["micro_" + name] = float(np.average(macro, weights=weights))
        micro = (macro[draws] * weights[draws]).sum(axis=1) / weights[draws].sum(axis=1)
        out["micro_" + name + "_ci95"] = np.quantile(micro, [0.025, 0.975]).tolist()
    return out


def combine(root, examples):
    rows = {r["example_id"]: r for r in read_rows(examples)}
    labels = json.loads((root / "judge/hidden_labels.json").read_text())
    learned = {}
    for folder, prefix in (
        ("learned", ""),
        ("learned_siblings", "sibling_train/"),
        ("learned_contextual", "contextual/"),
    ):
        path = root / folder / "pair_predictions.jsonl"
        if not path.exists():
            continue
        table = {(r["a_id"], r["b_id"]): r for r in read_rows(path)}
        learned[prefix] = table
    outputs, diagnostics = [], {"arms": {}, "fold_mismatches": []}
    for arm in ("within_run", "cross_run"):
        files = list((root / "judge/outputs" / arm).glob("*.json"))
        results = [json.loads(p.read_text()) for p in files]
        diagnostics["arms"][arm] = {
            "attempted": len(results),
            "valid": sum(r["valid"] for r in results),
            "reported_cost_usd": sum(r.get("cost_usd", 0) for r in results),
            "unknown_cost_calls": sum("cost_usd" not in r for r in results),
            "elapsed_call_seconds": sum(r["elapsed_seconds"] for r in results),
            "models": sorted({r["model_requested"] for r in results}),
            "invalid": [
                {"id": r["id"], "error": r.get("error")} for r in results if not r["valid"]
            ],
        }
    for row in labels:
        row = dict(row)
        row["probabilities"] = {"chance": 0.5}
        row["choices"] = {}
        a, b = rows[row["a_id"]], rows[row["b_id"]]
        later_a = a["first_submitted_at"] > b["first_submitted_at"]
        row["probabilities"]["later_recipe"] = float(later_a)
        row["scientist_model"] = a["scientist_model"]
        for prefix, table in learned.items():
            learned_row = table[(row["a_id"], row["b_id"])]
            if row["fold"] != learned_row["fold"]:
                raise ValueError("Learned and judge folds disagree")
            row["probabilities"].update(
                {
                    prefix + m: p
                    for m, p in learned_row["probabilities"].items()
                    if m != "random_choice"
                }
            )
        row["valid_arms"] = []
        for arm in ("within_run", "cross_run"):
            path = root / "judge/outputs" / arm / (row["id"] + ".json")
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            input_obj = json.loads(Path(result["input"]).read_text())
            if digest(input_obj) != result["input_sha256"]:
                raise ValueError("Frozen input hash differs from prediction metadata")
            if not result["valid"]:
                continue
            row["valid_arms"].append(arm)
            row["probabilities"]["frozen/" + arm] = result["p_a"]
            row["choices"]["frozen/" + arm] = result["choice_a"]
        if "cross_run" in row["valid_arms"]:
            row["probabilities"]["fixed_half_blend"] = 0.5 * (
                row["probabilities"]["frozen/cross_run"]
                + row["probabilities"]["fixed_recipe_numeric_logistic_C1"]
            )
        outputs.append(row)
    return outputs, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/analysis/rpm"))
    parser.add_argument(
        "--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl")
    )
    args = parser.parse_args()
    records, diagnostics = combine(args.root, args.examples)
    matched = [r for r in records if len(r["valid_arms"]) == 2]
    methods = (
        sorted(set.intersection(*(set(r["probabilities"]) for r in matched))) if matched else []
    )
    groups = {
        "matched_gap_01": matched,
        "matched_gap_02": [r for r in matched if r["gap"] >= 0.02],
        "matched_no_history": [r for r in matched if not r["history_count"]],
        "matched_with_history": [r for r in matched if r["history_count"]],
    }
    for scientist in sorted({r["scientist_model"] for r in records}):
        groups["matched_" + scientist] = [r for r in matched if r["scientist_model"] == scientist]
    summary = {
        name: {method: summarize(group, method) for method in methods}
        for name, group in groups.items()
    }
    comparisons = {
        name: [
            paired_difference(group, m, reference)
            for reference in ("frozen/within_run", "frozen/cross_run", "later_recipe")
            for m in methods
            if m != reference
        ]
        for name, group in groups.items()
    }
    diagnostics.update(
        total_pairs=len(records),
        matched_pairs=len(matched),
        matched_cells=len({r["cell_id"] for r in matched}),
        displayed_order={
            "swapped": sum(r["swapped"] for r in records),
            "not_swapped": sum(not r["swapped"] for r in records),
        },
    )
    args.root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.root / "comparison.json",
        {"diagnostics": diagnostics, "metrics": summary, "paired_comparisons": comparisons},
    )
    (args.root / "comparison_pairs.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    )
    print(json.dumps(diagnostics, indent=2))
    for method, metrics in summary["matched_gap_01"].items():
        print(
            f"{method}: macro={metrics['macro_accuracy']:.3f}, micro={metrics['micro_accuracy']:.3f}, regret={100 * metrics['macro_regret']:.2f}pp"
        )


if __name__ == "__main__":
    main()
