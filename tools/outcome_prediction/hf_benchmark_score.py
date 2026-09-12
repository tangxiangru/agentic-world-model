"""Score numeric predictions for the scripts/serving benchmark; never fit a model."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from tools.outcome_prediction.hf_benchmark import is_rate, read_jsonl


def predictions(path):
    result = {}
    for row in read_jsonl(path):
        if set(row) != {"example_id", "prediction"}:
            raise ValueError("Prediction rows must contain only example_id and prediction")
        if row["example_id"] in result:
            raise ValueError("Duplicate prediction ID")
        if not is_rate(row["prediction"]):
            raise ValueError("Predictions must be finite numbers in [0, 1]")
        result[row["example_id"]] = float(row["prediction"])
    return result


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] * (high - position) + values[high] * (position - low) if high != low else values[low]


def bootstrap(values, repeats=2000, seed=4129):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats)]
    return [quantile(means, 0.025), quantile(means, 0.975)]


def group_errors(rows, labels, predicted):
    errors = defaultdict(lambda: defaultdict(list))
    for row in rows:
        example = row["example_id"]
        if example in predicted:
            weights = row.get("weights_sha256") or row["checkpoint_id"]
            errors[row["group_id"]][weights].append(abs(predicted[example] - labels[example]) * 100)
    return {g: statistics.mean(statistics.mean(v) for v in weights.values())
            for g, weights in errors.items()}


def score(out, prediction_path, split="test", compare_path=None):
    out = Path(out)
    registry = [r for r in read_jsonl(out / "audit/registry.jsonl")
                if r["status"] == "eligible" and r["split"] == split]
    labels = {r["example_id"]: r["y"] for r in read_jsonl(out / "labels" / (split + ".jsonl"))}
    predicted = predictions(prediction_path)
    if set(predicted) - set(labels):
        raise ValueError("Predictions contain IDs outside this split")
    comparison = predictions(compare_path) if compare_path else None
    if comparison is not None and set(comparison) - set(labels):
        raise ValueError("Comparison contains IDs outside this split")
    results = {}
    for benchmark in sorted({r["benchmark"] for r in registry}):
        rows = [r for r in registry if r["benchmark"] == benchmark]
        available = [r for r in rows if r["example_id"] in predicted]
        complete = len(rows) == len(available)
        groups = group_errors(rows, labels, predicted)
        absolute = [abs(predicted[r["example_id"]] - labels[r["example_id"]]) * 100 for r in available]
        y = [labels[r["example_id"]] for r in available]
        sse = sum((predicted[r["example_id"]] - labels[r["example_id"]]) ** 2 for r in available)
        variance = sum((v - statistics.mean(y)) ** 2 for v in y) if y else 0
        item = {
            "expected_examples": len(rows), "predicted_examples": len(available),
            "coverage": len(available) / len(rows) if rows else None,
            "complete": complete, "groups_scored": len(groups),
            "primary_group_balanced_mae_pp": statistics.mean(groups.values()) if complete and groups else None,
            "primary_mae_95ci_pp": bootstrap(list(groups.values())) if complete else None,
            "available_case_cell_mae_pp": statistics.mean(absolute) if absolute else None,
            "available_case_r2": 1 - sse / variance if variance else None,
            "missing_ids": [r["example_id"] for r in rows if r["example_id"] not in predicted],
        }
        if comparison is not None:
            shared = [r for r in rows if r["example_id"] in predicted and r["example_id"] in comparison]
            a, b = group_errors(shared, labels, predicted), group_errors(shared, labels, comparison)
            differences = [a[g] - b[g] for g in sorted(a)]
            item["paired_comparison"] = {
                "shared_examples": len(shared), "coverage": len(shared) / len(rows),
                "groups": len(differences),
                "mae_difference_pp": statistics.mean(differences) if len(shared) == len(rows) and differences else None,
                "difference_95ci_pp": bootstrap(differences) if len(shared) == len(rows) else None,
                "direction": "negative favors first prediction file",
            }
        results[benchmark] = item
    return {"split": split, "metric": "mean over groups, then distinct weights, then serving cells",
            "uncertainty": "percentile bootstrap of entire groups; fixed benchmark questions",
            "benchmarks": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = score(args.benchmark_dir, args.predictions, args.split, args.compare)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
