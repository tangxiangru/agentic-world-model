"""Grouped, leakage-conscious benchmarks for checkpoint accuracy prediction.

Input is one JSON object per checkpoint. The extractor, rather than this script,
is responsible for ensuring that recipe fields contain only preexecution facts.
Numeric/categorical features must not contain run identities or observed outcomes.
All learned preprocessing is fit on training folds. Hyperparameters are fixed;
the OOF table is exploratory model comparison, not a fresh confirmatory test.

Example:
    python tools/outcome_prediction/benchmark.py --examples examples.jsonl \
        --output-dir results --permutations 50

Dependencies: numpy, scipy, scikit-learn. Accuracy fields must be fractions.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

METHODS = [
    "train_mean",
    "train_median",
    "scientist_only_nuisance",
    "method_only",
    "numeric_ridge",
    "numeric_extra_trees",
    "structured_ridge",
    "recipe_tfidf_ridge",
    "recipe_tfidf_neighbors",
    "lineage_tfidf_ridge",
    "lineage_tfidf_neighbors",
    "recipe_numeric_ridge",
    "lineage_numeric_ridge",
    "comparator_accuracy",
    "oracle_parent_accuracy",
    "postexecution_local_accuracy",
]
EXPLORATORY_METHODS = ["plan_tfidf_ridge", "lineage_plan_tfidf_ridge"]
NONRECIPE_METHODS = {
    "scientist_only_nuisance",
    "oracle_parent_accuracy",
    "postexecution_local_accuracy",
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(clean_json(row), sort_keys=True) + "\n")


def load_examples(path: Path, require_complete: bool) -> tuple[list[dict], dict]:
    source_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = []
    exclusions = Counter()
    for source_index, row in enumerate(source_rows):
        if row.get("y") is None:
            exclusions["unlabeled"] += 1
        elif not row.get("eligible", False):
            exclusions["ineligible"] += 1
        elif require_complete and not row.get("lineage_complete", False):
            exclusions["incomplete_lineage"] += 1
        else:
            if not np.isfinite(float(row["y"])) or not 0 <= float(row["y"]) <= 1:
                raise ValueError(f"Accuracy outside [0,1]: {row.get('example_id')}")
            row["_source_index"] = source_index
            rows.append(row)
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example_id values; checkpoint deduplication is required")
    return rows, {"source_rows": len(source_rows), "exclusions": dict(exclusions)}


def split_groups(rows: list[dict], n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Randomize cell assignment independently of outcomes, balancing row counts."""
    groups = np.array([str(row["cell_id"]) for row in rows])
    unique, counts = np.unique(groups, return_counts=True)
    n_folds = min(n_folds, len(unique))
    if n_folds < 2:
        raise ValueError("Grouped CV requires at least two eligible cells")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    # Largest cells first; random ordering breaks equal-size ties reproducibly.
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_cells: list[list[str]] = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)
    for index in order:
        candidate = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold = int(rng.choice(candidate))
        fold_cells[fold].append(unique[index])
        fold_sizes[fold] += counts[index]
    result = []
    for cells in fold_cells:
        test_mask = np.isin(groups, cells)
        result.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return result


def documents(rows: list[dict], key: str) -> list[str]:
    # Sentinel guarantees a usable vocabulary even for entirely missing recipes.
    return ["recipepresent " + str(row.get(key) or "") for row in rows]


def text_features(train: list[dict], test: list[dict], key: str) -> tuple[Any, Any]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=12000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w.]+\b",
    )
    return vectorizer.fit_transform(documents(train, key)), vectorizer.transform(
        documents(test, key)
    )


def numeric_features(train: list[dict], test: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted({key for row in train for key in row.get("numeric_features", {})})
    if not keys:
        return np.zeros((len(train), 1)), np.zeros((len(test), 1))

    def array(rows: list[dict]) -> np.ndarray:
        result = np.array(
            [[row.get("numeric_features", {}).get(key, np.nan) for key in keys] for row in rows],
            dtype=float,
        )
        result[~np.isfinite(result)] = np.nan
        return result

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(array(train))
    x_test = imputer.transform(array(test))
    if not x_train.shape[1]:
        return np.zeros((len(train), 1)), np.zeros((len(test), 1))
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def categorical_features(train: list[dict], test: list[dict]) -> tuple[Any, Any]:
    vectorizer = DictVectorizer(sparse=True)

    def values(rows: list[dict]) -> list[dict]:
        return [
            {
                "constant": "present",
                **{
                    str(key): str(value)
                    for key, value in row.get("categorical_features", {}).items()
                    if value is not None
                },
            }
            for row in rows
        ]

    return vectorizer.fit_transform(values(train)), vectorizer.transform(values(test))


def ridge_prediction(x_train: Any, x_test: Any, y_train: np.ndarray) -> np.ndarray:
    return Ridge(alpha=10.0, solver="lsqr").fit(x_train, y_train).predict(x_test)


def neighbor_prediction(x_train: Any, x_test: Any, y_train: np.ndarray) -> np.ndarray:
    similarities = (x_test @ x_train.T).toarray()
    result = np.full(x_test.shape[0], float(np.mean(y_train)))
    for index, similarity in enumerate(similarities):
        selected = np.argsort(-similarity, kind="stable")[: min(5, len(y_train))]
        weights = np.maximum(similarity[selected], 0) ** 2
        if weights.sum() > 1e-12:
            result[index] = np.average(y_train[selected], weights=weights)
    return result


def grouped_means(train: list[dict], test: list[dict], y_train: np.ndarray, key: str) -> np.ndarray:
    def category(row: dict) -> str:
        if key == "method":
            features = row.get("categorical_features", {})
            return str(features.get("method", features.get("training_method", "unknown")))
        return str(row.get(key, "unknown"))

    labels: dict[str, list[float]] = defaultdict(list)
    for row, y in zip(train, y_train):
        labels[category(row)].append(float(y))
    mean = float(np.mean(y_train))
    # Fixed three-observation prior prevents singleton categories from dominating.
    return np.array(
        [(sum(labels[category(row)]) + 3 * mean) / (len(labels[category(row)]) + 3) for row in test]
    )


def predict_methods(train: list[dict], test: list[dict], seed: int) -> dict[str, np.ndarray]:
    y_train = np.array([row["y"] for row in train], dtype=float)
    mean = float(np.mean(y_train))
    result = {
        "train_mean": np.full(len(test), mean),
        "train_median": np.full(len(test), np.median(y_train)),
        "scientist_only_nuisance": grouped_means(train, test, y_train, "scientist_model"),
        "method_only": grouped_means(train, test, y_train, "method"),
    }
    nx_train, nx_test = numeric_features(train, test)
    cx_train, cx_test = categorical_features(train, test)
    result["numeric_ridge"] = ridge_prediction(nx_train, nx_test, y_train)
    result["numeric_extra_trees"] = (
        ExtraTreesRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=4,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        )
        .fit(nx_train, y_train)
        .predict(nx_test)
    )
    result["structured_ridge"] = ridge_prediction(
        sparse.hstack([sparse.csr_matrix(nx_train), cx_train], format="csr"),
        sparse.hstack([sparse.csr_matrix(nx_test), cx_test], format="csr"),
        y_train,
    )
    for prefix, key in [("recipe", "recipe_text"), ("lineage", "lineage_text")]:
        tx_train, tx_test = text_features(train, test, key)
        result[f"{prefix}_tfidf_ridge"] = ridge_prediction(tx_train, tx_test, y_train)
        result[f"{prefix}_tfidf_neighbors"] = neighbor_prediction(tx_train, tx_test, y_train)
        result[f"{prefix}_numeric_ridge"] = ridge_prediction(
            sparse.hstack([tx_train, sparse.csr_matrix(nx_train)], format="csr"),
            sparse.hstack([tx_test, sparse.csr_matrix(nx_test)], format="csr"),
            y_train,
        )
    for prefix, key in [("plan", "plan_text"), ("lineage_plan", "lineage_plan_text")]:
        if any(row.get(key) for row in train + test):
            tx_train, tx_test = text_features(train, test, key)
            result[f"{prefix}_tfidf_ridge"] = ridge_prediction(tx_train, tx_test, y_train)
    for method, key in [
        ("comparator_accuracy", "comparator_accuracy"),
        ("oracle_parent_accuracy", "parent_accuracy"),
        ("postexecution_local_accuracy", "local_accuracy"),
    ]:
        result[method] = np.array(
            [
                float(row[key])
                if row.get(key) is not None and np.isfinite(float(row[key]))
                else mean
                for row in test
            ]
        )
    return {key: np.clip(value, 0, 1) for key, value in result.items()}


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    residual = y - pred
    total_ss = float(np.sum((y - y.mean()) ** 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation = float(spearmanr(y, pred).statistic) if len(y) > 1 else np.nan
    return {
        "n": len(y),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1 - float(np.sum(residual**2)) / total_ss if total_ss > 0 else np.nan,
        "spearman": correlation,
    }


def assess(rows: list[dict], predictions: dict[str, np.ndarray], seed: int) -> dict:
    y = np.array([row["y"] for row in rows], dtype=float)
    groups = np.array([str(row["cell_id"]) for row in rows])
    cells = np.unique(groups)
    cell_indices = {cell: np.flatnonzero(groups == cell) for cell in cells}
    boot = np.random.default_rng(seed).integers(0, len(cells), size=(5000, len(cells)))
    baseline_loss = np.array(
        [np.mean(np.abs(y[ix] - predictions["train_mean"][ix])) for ix in cell_indices.values()]
    )
    methods = {}
    for method, pred in predictions.items():
        item = regression_metrics(y, pred)
        cell_loss, concordances, regrets = [], [], []
        concordant = comparable = 0.0
        per_cell = {}
        for cell, ix in cell_indices.items():
            cy, cp = y[ix], pred[ix]
            loss = float(np.mean(np.abs(cy - cp)))
            cell_loss.append(loss)
            correct, count = 0.0, 0
            for left in range(len(ix)):
                for right in range(left + 1, len(ix)):
                    difference = cy[left] - cy[right]
                    if abs(difference) < 0.02:
                        continue
                    predicted = cp[left] - cp[right]
                    correct += 0.5 if abs(predicted) < 1e-12 else float(difference * predicted > 0)
                    count += 1
            concordant += correct
            comparable += count
            if count:
                concordances.append(correct / count)
            # Uniformly break prediction ties; otherwise a constant predictor
            # would get arbitrary credit from file ordering.
            selected = np.isclose(cp, np.max(cp), atol=1e-12, rtol=0)
            regret = float(cy.max() - cy[selected].mean())
            if len(ix) > 1:
                regrets.append(regret)
            per_cell[cell] = {
                "n": len(ix),
                "mae": loss,
                "comparable_pairs": count,
                "pairwise_concordance": correct / count if count else None,
                "selection_regret": regret,
            }
        cell_loss = np.array(cell_loss)
        improvements = baseline_loss - cell_loss
        boot_improvements = improvements[boot].mean(axis=1)
        item.update(
            {
                "cell_weighted_mae": float(cell_loss.mean()),
                "cell_weighted_mae_ci95": np.quantile(cell_loss[boot].mean(axis=1), [0.025, 0.975]),
                "mae_improvement_vs_mean": float(improvements.mean()),
                "mae_improvement_vs_mean_ci95": np.quantile(boot_improvements, [0.025, 0.975]),
                "cells_with_lower_mae_than_mean": int(np.sum(improvements > 1e-12)),
                "cells_with_higher_mae_than_mean": int(np.sum(improvements < -1e-12)),
                "pairwise_concordance": concordant / comparable if comparable else None,
                "cell_weighted_pairwise_concordance": float(np.mean(concordances))
                if concordances
                else None,
                "comparable_pairs": int(comparable),
                "mean_selection_regret": float(np.mean(regrets)) if regrets else None,
                "per_cell": per_cell,
            }
        )
        methods[method] = item
    selection_baselines: dict[str, list[float]] = defaultdict(list)
    for ix in cell_indices.values():
        if len(ix) < 2:
            continue
        cy = y[ix]
        chronology = sorted(ix, key=lambda index: rows[index]["_source_index"])
        selection_baselines["uniform_random"].append(float(cy.max() - cy.mean()))
        selection_baselines["first_in_source_order"].append(float(cy.max() - y[chronology[0]]))
        selection_baselines["last_in_source_order"].append(float(cy.max() - y[chronology[-1]]))
        selection_baselines["oracle"].append(0.0)
    return {
        "n_examples": len(rows),
        "n_cells": len(cells),
        "target": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        },
        "methods": methods,
        "selection_baselines": {
            key: float(np.mean(value)) for key, value in selection_baselines.items()
        },
        "n_multicheckpoint_cells": len(selection_baselines.get("uniform_random", [])),
    }


def oof_evaluate(
    rows: list[dict], folds: list[tuple[np.ndarray, np.ndarray]], seed: int
) -> tuple[dict, list[dict], list[dict]]:
    predictions: dict[str, np.ndarray] = {}
    assignments = []
    row_folds = np.full(len(rows), -1)
    for fold, (train_indices, test_indices) in enumerate(folds):
        train = [rows[index] for index in train_indices]
        test = [rows[index] for index in test_indices]
        train_cells = sorted({row["cell_id"] for row in train})
        test_cells = sorted({row["cell_id"] for row in test})
        assert set(train_cells).isdisjoint(test_cells)
        assignments.append({"fold": fold, "train_cells": train_cells, "test_cells": test_cells})
        row_folds[test_indices] = fold
        for method, values in predict_methods(train, test, seed).items():
            predictions.setdefault(method, np.full(len(rows), np.nan))[test_indices] = values
        print(f"Fold {fold + 1}/{len(folds)}: {len(train)} train, {len(test)} test", flush=True)
    assert np.all(row_folds >= 0)
    for method, values in predictions.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Missing or nonfinite OOF predictions for {method}")
    records = [
        {
            "example_id": row["example_id"],
            "cell_id": row["cell_id"],
            "scientist_model": row.get("scientist_model"),
            "card_id": row.get("card_id"),
            "fold": int(row_folds[index]),
            "y": row["y"],
            "predictions": {method: float(values[index]) for method, values in predictions.items()},
        }
        for index, row in enumerate(rows)
    ]
    return assess(rows, predictions, seed), records, assignments


def transfer_evaluate(rows: list[dict], seed: int) -> tuple[dict, list[dict]]:
    scientists = sorted({str(row.get("scientist_model", "unknown")) for row in rows})
    results, records = {}, []
    if len(scientists) < 2:
        return {"skipped": "Only one scientist model is represented"}, records
    for scientist in scientists:
        train = [row for row in rows if str(row.get("scientist_model", "unknown")) == scientist]
        test = [row for row in rows if str(row.get("scientist_model", "unknown")) != scientist]
        if {row["cell_id"] for row in train} & {row["cell_id"] for row in test}:
            raise ValueError("Scientist transfer would split a cell across train and test")
        pred = predict_methods(train, test, seed)
        label = f"train_{scientist}"
        results[label] = assess(test, pred, seed)
        results[label]["train_n"] = len(train)
        results[label]["train_cells"] = sorted({row["cell_id"] for row in train})
        for index, row in enumerate(test):
            records.append(
                {
                    "split": label,
                    "example_id": row["example_id"],
                    "cell_id": row["cell_id"],
                    "y": row["y"],
                    "predictions": {
                        method: float(values[index]) for method, values in pred.items()
                    },
                }
            )
    return results, records


def permutation_control(
    rows: list[dict], folds: list[tuple[np.ndarray, np.ndarray]], count: int, seed: int
) -> dict:
    """Label shuffle is a negative control, not a clustered significance test."""
    y = np.array([row["y"] for row in rows], dtype=float)
    cached = []
    for train_ix, test_ix in folds:
        tx_train, tx_test = text_features(
            [rows[index] for index in train_ix],
            [rows[index] for index in test_ix],
            "lineage_text",
        )
        cached.append((train_ix, test_ix, tx_train, tx_test))
    rng = np.random.default_rng(seed + 1000)
    maes, mean_maes = [], []
    for _ in range(count):
        shuffled = rng.permutation(y)
        pred, baseline = np.zeros(len(rows)), np.zeros(len(rows))
        for train_ix, test_ix, tx_train, tx_test in cached:
            pred[test_ix] = np.clip(ridge_prediction(tx_train, tx_test, shuffled[train_ix]), 0, 1)
            baseline[test_ix] = shuffled[train_ix].mean()
        maes.append(float(np.mean(np.abs(pred - shuffled))))
        mean_maes.append(float(np.mean(np.abs(baseline - shuffled))))
    return {
        "method": "lineage_tfidf_ridge",
        "repetitions": count,
        "warning": "Global label shuffling breaks cell dependence; this is a software/signal negative control, not an inferential p-value.",
        "shuffled_mae": maes,
        "shuffled_mean_baseline_mae": mean_maes,
        "mean_shuffled_mae": float(np.mean(maes)),
        "mean_shuffled_improvement_vs_mean": float(np.mean(np.array(mean_maes) - maes)),
    }


def table(metrics: dict) -> str:
    lines = [
        "| Method | MAE (pp) | RMSE (pp) | R² | Spearman | Cell MAE gain vs mean (pp), 95% CI | Pair concordance | Selection regret (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value: Any, scale: float = 1) -> str:
        return "—" if value is None or not np.isfinite(value) else f"{value * scale:.3f}"

    for method, item in metrics["methods"].items():
        lower, upper = item["mae_improvement_vs_mean_ci95"]
        interval = f"{number(item['mae_improvement_vs_mean'], 100)} [{number(lower, 100)}, {number(upper, 100)}]"
        lines.append(
            f"| {method} | {number(item['mae'], 100)} | {number(item['rmse'], 100)} | "
            f"{number(item['r2'])} | {number(item['spearman'])} | {interval} | "
            f"{number(item['pairwise_concordance'])} | {number(item['mean_selection_regret'], 100)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--require-complete-lineage", action="store_true")
    parser.add_argument(
        "--split-json",
        type=Path,
        help='Optional fixed split: {"train_cells": [...], "test_cells": [...]}',
    )
    args = parser.parse_args()
    rows, audit = load_examples(args.examples, args.require_complete_lineage)
    if not rows:
        raise ValueError("No eligible labeled examples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    notes = {
        "units": "Accuracies and errors are fractions; displayed table errors are percentage points.",
        "input_contract": "Recipe, lineage, numeric and categorical fields must contain only preexecution facts; extractor is responsible for this audit.",
        "hyperparameters": "Fixed, no test tuning: TFIDF word 1-2 grams max 12000; ridge alpha=10; cosine neighbors k=5 squared weights; ExtraTrees 300 depth=4 leaf=4; category mean prior n=3.",
        "split": "Cells are wholly held out; folds use seeded size balancing independent of labels.",
        "confidence_intervals": "5000 paired cell bootstraps of OOF errors; descriptive, not corrected for crossfold training dependence or multiple model comparisons.",
        "selection": "Choose a checkpoint among observed eligible checkpoints within a held-out cell; average ties uniformly. This is retrospective, not an online policy evaluation.",
        "pairwise": "Within-cell pairs with absolute true accuracy difference >=0.02; prediction ties get 0.5 credit.",
        "nuisance_and_ceiling_arms": sorted(NONRECIPE_METHODS),
        "exploratory_prose_arms": EXPLORATORY_METHODS,
        "missing_score_baselines": "Missing comparator/parent/local scores fall back to the training-fold mean.",
        "recipe_only_scope": "Only surviving, labeled, eligible checkpoints; failed experiments and missing official scores are not assigned zero accuracy.",
    }
    coverage = {
        key: sum(row.get(key) is not None for row in rows)
        for key in [
            "comparator_accuracy",
            "parent_accuracy",
            "local_accuracy",
        ]
    }
    if args.split_json:
        split = json.loads(args.split_json.read_text())
        train_cells, test_cells = set(split["train_cells"]), set(split["test_cells"])
        if train_cells & test_cells:
            raise ValueError("Explicit train/test cell overlap")
        known_cells = {row["cell_id"] for row in rows}
        unknown = (train_cells | test_cells) - known_cells
        if unknown:
            raise ValueError(
                f"Explicit split contains cells absent from eligible cohort: {sorted(unknown)}"
            )
        train = [row for row in rows if row["cell_id"] in train_cells]
        test = [row for row in rows if row["cell_id"] in test_cells]
        if not train or not test:
            raise ValueError("Explicit split has an empty train or test partition")
        pred = predict_methods(train, test, args.seed)
        result = assess(test, pred, args.seed)
        result.update({"audit": audit, "notes": notes, "train_n": len(train), "coverage": coverage})
        records = [
            {
                "example_id": row["example_id"],
                "cell_id": row["cell_id"],
                "y": row["y"],
                "predictions": {method: float(values[index]) for method, values in pred.items()},
            }
            for index, row in enumerate(test)
        ]
        write_json(args.output_dir / "fold_assignments.json", split)
    else:
        folds = split_groups(rows, args.folds, args.seed)
        result, records, assignments = oof_evaluate(rows, folds, args.seed)
        transfer, transfer_records = transfer_evaluate(rows, args.seed)
        result.update(
            {"scientist_transfer": transfer, "audit": audit, "notes": notes, "coverage": coverage}
        )
        if args.permutations > 0:
            result["permutation_negative_control"] = permutation_control(
                rows, folds, args.permutations, args.seed
            )
        write_json(args.output_dir / "fold_assignments.json", assignments)
        write_jsonl(args.output_dir / "scientist_transfer_predictions.jsonl", transfer_records)
    result["configuration"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    result["package_versions"] = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    write_json(args.output_dir / "metrics.json", result)
    write_jsonl(args.output_dir / "predictions.jsonl", records)
    rendered = table(result)
    (args.output_dir / "table.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
