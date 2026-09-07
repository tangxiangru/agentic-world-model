"""Nested regularization selection, lineage ablations, and scientist controls."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from benchmark import assess, clean_json, load_examples, oof_evaluate, split_groups, text_features
from sklearn.linear_model import Ridge


def selected_ridge(train, test, field, seed):
    candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    folds = split_groups(train, 4, seed)
    losses = {a: [] for a in candidates}
    for train_idx, valid_idx in folds:
        inner_train = [train[i] for i in train_idx]
        valid = [train[i] for i in valid_idx]
        x, z = text_features(inner_train, valid, field)
        y = np.array([r["y"] for r in inner_train])
        truth = np.array([r["y"] for r in valid])
        groups = np.array([r["cell_id"] for r in valid])
        for alpha in candidates:
            pred = np.clip(Ridge(alpha=alpha, solver="lsqr").fit(x, y).predict(z), 0, 1)
            losses[alpha].extend(
                np.mean(np.abs(pred[groups == c] - truth[groups == c])) for c in np.unique(groups)
            )
    alpha = min(candidates, key=lambda a: np.mean(losses[a]))
    x, z = text_features(train, test, field)
    y = np.array([r["y"] for r in train])
    return np.clip(Ridge(alpha=alpha, solver="lsqr").fit(x, y).predict(z), 0, 1), alpha


def no_style_recipe(recipe):
    recipe = copy.deepcopy(recipe)
    recipe.pop("framework", None)
    recipe["hyperparameters"].pop("seed", None)
    # Keep physical precision/optimizer settings: these affect training.
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/analysis/outcome_prediction/sensitivity")
    )
    args = parser.parse_args()
    rows, _ = load_examples(args.examples, False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["recipe_no_package_seed"] = json.dumps(no_style_recipe(row["recipe"]), sort_keys=True)
    fields = ["recipe_text", "lineage_text", "recipe_no_package_seed"]
    predictions = {"train_mean": np.zeros(len(rows))}
    for field in fields:
        predictions[field + "_nested"] = np.zeros(len(rows))
    choices = []
    for fold, (train_idx, test_idx) in enumerate(split_groups(rows, 8, 42)):
        train, test = [rows[i] for i in train_idx], [rows[i] for i in test_idx]
        predictions["train_mean"][test_idx] = np.mean([r["y"] for r in train])
        for field in fields:
            pred, alpha = selected_ridge(train, test, field, 42 + fold)
            predictions[field + "_nested"][test_idx] = pred
            choices.append({"fold": fold, "field": field, "alpha": alpha})
        print(f"Nested fold {fold + 1}/8", flush=True)
    result = {
        "nested": assess(rows, predictions, 42),
        "alpha_choices": choices,
        "within_scientist": {},
        "within_scientist_nested": {},
        "nested_scientist_transfer": {},
    }
    records = [
        {
            "example_id": r["example_id"],
            "cell_id": r["cell_id"],
            "y": r["y"],
            "predictions": {k: float(v[i]) for k, v in predictions.items()},
        }
        for i, r in enumerate(rows)
    ]
    (args.output_dir / "nested_predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    for scientist in sorted({r["scientist_model"] for r in rows}):
        subset = [r for r in rows if r["scientist_model"] == scientist]
        metrics, preds, _folds = oof_evaluate(subset, split_groups(subset, 8, 42), 42)
        result["within_scientist"][scientist] = metrics
        (args.output_dir / (scientist + "_predictions.jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in preds)
        )
        conditional_preds = {
            "train_mean": np.zeros(len(subset)),
            "recipe_nested": np.zeros(len(subset)),
            "lineage_nested": np.zeros(len(subset)),
        }
        for fold, (tr, te) in enumerate(split_groups(subset, 8, 42)):
            train, test = [subset[i] for i in tr], [subset[i] for i in te]
            conditional_preds["train_mean"][te] = np.mean([r["y"] for r in train])
            for key, field in [
                ("recipe_nested", "recipe_text"),
                ("lineage_nested", "lineage_text"),
            ]:
                conditional_preds[key][te], _ = selected_ridge(train, test, field, 42 + fold)
        result["within_scientist_nested"][scientist] = assess(subset, conditional_preds, 42)
        other = [r for r in rows if r["scientist_model"] != scientist]
        transfer_preds = {"train_mean": np.full(len(other), np.mean([r["y"] for r in subset]))}
        for key, field in [("recipe_nested", "recipe_text"), ("lineage_nested", "lineage_text")]:
            transfer_preds[key], _ = selected_ridge(subset, other, field, 42)
        result["nested_scientist_transfer"][scientist] = assess(other, transfer_preds, 42)
    complete = [r for r in rows if r["lineage_complete"]]
    result["complete_lineage"], _, _ = oof_evaluate(complete, split_groups(complete, 8, 42), 42)
    result["notes"] = [
        "Alpha selected only inside each outer training fold by four-fold cell-grouped, cell-weighted MAE.",
        "Package/seed ablation removes package/version and seed features, retains actual physical precision and optimizer settings.",
        "Within-scientist CV holds out entire runs but fits and evaluates on one scientist model only.",
        "All comparisons remain exploratory; confidence intervals are descriptive and not multiple-comparison corrected.",
    ]
    (args.output_dir / "metrics.json").write_text(json.dumps(clean_json(result), indent=2) + "\n")
    for group, metrics in [("nested", result["nested"]), *result["within_scientist"].items()]:
        print(group)
        for name, m in metrics["methods"].items():
            if name in {
                "train_mean",
                "train_median",
                "numeric_extra_trees",
                "recipe_tfidf_ridge",
                "lineage_tfidf_ridge",
            } or name.endswith("_nested"):
                print(
                    name,
                    "MAE pp",
                    round(m["mae"] * 100, 3),
                    "R2",
                    round(m["r2"], 3),
                    "pair",
                    m["pairwise_concordance"],
                )


if __name__ == "__main__":
    main()
