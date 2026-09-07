"""Aggregate independent blinded sessions on disjoint held-out scientist runs."""

import json
from pathlib import Path

import numpy as np
from benchmark import assess, clean_json


def main():
    root = Path("data/analysis/outcome_prediction")
    dirs = [root / "icl", root / "icl_fold1", root / "icl_fold2"]
    rows, seen, folds = [], set(), []
    for directory in dirs:
        split = json.loads((directory / "split.json").read_text())
        assert seen.isdisjoint(split["test_cells"])
        seen.update(split["test_cells"])
        part = [
            json.loads(s) for s in (directory / "scored_predictions.jsonl").read_text().splitlines()
        ]
        for row in part:
            row["_source_index"] = len(rows)
            row["blind_session"] = directory.name
            rows.append(row)
        folds.append({"session": directory.name, "n": len(part), **split})
    assert len({r["example_id"] for r in rows}) == len(rows)
    methods = set(rows[0]["predictions"])
    assert all(set(r["predictions"]) == methods for r in rows)
    preds = {m: np.array([r["predictions"][m] for r in rows]) for m in sorted(methods)}
    result = assess(rows, preds, 42)
    result["folds"] = folds
    result["notes"] = [
        "Three fresh-context blinded model sessions; each trains in context on 24 whole runs and predicts 8 disjoint runs.",
        "107 unique checkpoints from 24 held-out runs; the remaining 8 runs contribute training examples only in this exercise.",
        "The fourth prepared split was not run because the session reached its agent-thread limit; no outcomes were used to select the three completed splits.",
        "Zero-shot files were frozen before each session read its training labels; sessions received no evaluation feedback.",
        "Same-split statistical baselines are included. Do not compare these MAEs directly with the main 146-example 8-fold table.",
        "One generation per prompt/split; model-session identity is inherited rather than a pinned external API model.",
    ]
    paired = {}
    cells = sorted({r["cell_id"] for r in rows})
    for comparator in ("zero_shot", "scientist_only_nuisance", "recipe_tfidf_neighbors"):
        differences = np.array(
            [
                result["methods"][comparator]["per_cell"][c]["mae"]
                - result["methods"]["few_shot"]["per_cell"][c]["mae"]
                for c in cells
            ]
        )
        boot = np.random.default_rng(42).choice(differences, size=(5000, len(cells))).mean(axis=1)
        paired[comparator] = {
            "cell_weighted_mae_gain": differences.mean(),
            "ci95": np.quantile(boot, [0.025, 0.975]),
        }
    result["few_shot_paired_improvements"] = paired
    out = root / "icl_combined"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(clean_json(result), indent=2) + "\n")
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for method in (
        "zero_shot",
        "few_shot",
        "train_mean",
        "scientist_only_nuisance",
        "recipe_tfidf_neighbors",
        "lineage_tfidf_ridge",
    ):
        print(
            method,
            {
                k: result["methods"][method][k]
                for k in ("mae", "rmse", "r2", "pairwise_concordance", "mean_selection_regret")
            },
        )
    print("paired", clean_json(paired))


if __name__ == "__main__":
    main()
