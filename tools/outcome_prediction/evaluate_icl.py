"""Score frozen blinded predictions; this process may read the hidden labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark import assess, clean_json


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("data/analysis/outcome_prediction/icl")
    )
    args = parser.parse_args()
    root = args.directory
    hidden = read_jsonl(root / "hidden_labels.jsonl")
    tests = read_jsonl(root / "test.jsonl")
    training = read_jsonl(root / "train.jsonl")
    split = json.loads((root / "split.json").read_text())
    assert set(split["train_cells"]).isdisjoint(split["test_cells"])
    assert {r["id"] for r in tests} == {r["id"] for r in hidden}
    assert all(set(r) == {"id", "recipe_sequence"} for r in tests)
    prediction_sets = {}
    hashes = {}
    for mode in ("zero_shot", "few_shot"):
        path = root / (mode + "_predictions.jsonl")
        raw = read_jsonl(path)
        assert len(raw) == len(hidden)
        predictions = {r["id"]: float(r["prediction"]) for r in raw}
        assert set(predictions) == {r["id"] for r in hidden}
        assert all(np.isfinite(v) and 0 <= v <= 1 for v in predictions.values())
        prediction_sets[mode] = np.array([predictions[r["id"]] for r in hidden])
        hashes[mode] = hashlib.sha256(path.read_bytes()).hexdigest()
    tabular = {r["example_id"]: r for r in read_jsonl(root / "tabular_holdout/predictions.jsonl")}
    assert set(tabular) == {r["example_id"] for r in hidden}
    for method in next(iter(tabular.values()))["predictions"]:
        prediction_sets[method] = np.array(
            [tabular[r["example_id"]]["predictions"][method] for r in hidden]
        )
    rows = [{**r, "_source_index": i} for i, r in enumerate(hidden)]
    result = assess(rows, prediction_sets, 42)
    groups = sorted({r["cell_id"] for r in rows})
    y = np.array([r["y"] for r in rows])
    improvements = []
    for group in groups:
        mask = np.array([r["cell_id"] == group for r in rows])
        improvements.append(
            np.mean(np.abs(y[mask] - prediction_sets["zero_shot"][mask]))
            - np.mean(np.abs(y[mask] - prediction_sets["few_shot"][mask]))
        )
    boot = np.random.default_rng(42).choice(improvements, size=(5000, len(groups))).mean(axis=1)
    result["few_shot_gain_vs_zero_shot"] = {
        "cell_weighted_mae_gain": np.mean(improvements),
        "ci95": np.quantile(boot, [0.025, 0.975]),
    }
    result["protocol"] = {
        "training_examples": len(training),
        "training_cells": len(split["train_cells"]),
        "heldout_cells": len(split["test_cells"]),
        "predictor": "Fresh-context Codex subagent inheriting the session model; no external API, model fitting, or hidden labels.",
        "sequence": "Frozen zero-shot predictions before opening labeled training bank; then frozen in-context predictions on same test set.",
        "limit": "One model session and one split; no claim of repeated-run stability, pinned API model identity, or population-level significance.",
        "artifact_hashes": hashes,
    }
    (root / "metrics.json").write_text(json.dumps(clean_json(result), indent=2) + "\n")
    scored = [
        {**r, "predictions": {k: float(v[i]) for k, v in prediction_sets.items()}}
        for i, r in enumerate(hidden)
    ]
    (root / "scored_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored))
    for name in (
        "zero_shot",
        "few_shot",
        "train_mean",
        "scientist_only_nuisance",
        "recipe_tfidf_neighbors",
        "lineage_tfidf_ridge",
    ):
        m = result["methods"][name]
        print(
            name,
            "MAE pp",
            round(m["mae"] * 100, 3),
            "R2",
            round(m["r2"], 3),
            "pairwise",
            m["pairwise_concordance"],
        )
    print("Few shot versus zero", clean_json(result["few_shot_gain_vs_zero_shot"]))


if __name__ == "__main__":
    main()
