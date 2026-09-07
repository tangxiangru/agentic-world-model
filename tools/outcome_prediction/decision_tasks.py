"""Grouped tests of threshold decisions and parent-conditioned accuracy changes.

Uses only eligible checkpoint examples and the same full-cohort cell folds as
benchmark.py. Current-checkpoint local scores are never input features. The
official parent accuracy is only used to define the retrospective improvement
label; pre-plan parent_local_accuracy is the only parent score used as input.

These are exploratory alternatives examined on the same small dataset, not
independent confirmation of a selected result. Fixed hyperparameters and fixed
classification probability threshold 0.5 avoid tuning on OOF outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from benchmark import (
    categorical_features,
    clean_json,
    load_examples,
    numeric_features,
    regression_metrics,
    ridge_prediction,
    split_groups,
    text_features,
    write_json,
    write_jsonl,
)
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

SAFE_CATEGORIES = {"method", "parent_kind", "peft", "precision", "scheduler"}


def feature_rows(rows: list[dict], condition_key: str | None) -> list[dict]:
    result = []
    for row in rows:
        copy = deepcopy(row)
        copy["categorical_features"] = {
            key: value
            for key, value in row.get("categorical_features", {}).items()
            if key in SAFE_CATEGORIES
        }
        if condition_key:
            copy["numeric_features"]["known_" + condition_key] = row[condition_key]
        result.append(copy)
    return result


def structured_features(train: list[dict], test: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    nx_train, nx_test = numeric_features(train, test)
    cx_train, cx_test = categorical_features(train, test)
    return (
        np.hstack([nx_train, cx_train.toarray()]),
        np.hstack([nx_test, cx_test.toarray()]),
    )


def scientist_probabilities(train: list[dict], test: list[dict], y: np.ndarray) -> np.ndarray:
    categories: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(train, y):
        categories[row["scientist_model"]].append(float(value))
    mean = float(y.mean())
    return np.array(
        [
            (sum(categories[row["scientist_model"]]) + 3 * mean)
            / (len(categories[row["scientist_model"]]) + 3)
            for row in test
        ]
    )


def fit_classifier(
    kind: str, x_train: np.ndarray, x_test: np.ndarray, y: np.ndarray, seed: int
) -> np.ndarray:
    if len(np.unique(y)) < 2:
        return np.full(len(x_test), float(y.mean()))
    if kind == "logistic":
        model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    else:
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        )
    return model.fit(x_train, y).predict_proba(x_test)[:, 1]


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    predicted = p >= 0.5
    both = len(np.unique(y)) == 2
    bins = []
    bin_numbers = np.minimum((p * 5).astype(int), 4)
    ece = 0.0
    for index in range(5):
        selected = bin_numbers == index
        if not selected.any():
            continue
        mean_p, frequency = float(p[selected].mean()), float(y[selected].mean())
        count = int(selected.sum())
        ece += count / len(y) * abs(mean_p - frequency)
        bins.append(
            {
                "lower": index / 5,
                "upper": (index + 1) / 5,
                "n": count,
                "mean_probability": mean_p,
                "observed_frequency": frequency,
            }
        )
    return {
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, predicted)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_5_equal_width_bins": ece,
        "calibration_bins": bins,
        "predicted_positive_count": int(predicted.sum()),
        "true_positive_count": int(np.sum(predicted & (y == 1))),
        "false_positive_count": int(np.sum(predicted & (y == 0))),
        "true_negative_count": int(np.sum(~predicted & (y == 0))),
        "false_negative_count": int(np.sum(~predicted & (y == 1))),
    }


def evaluate_classification(
    rows: list[dict], y: np.ndarray, predictions: dict[str, np.ndarray], seed: int
) -> dict:
    cells = sorted({row["cell_id"] for row in rows})
    indices = [
        np.array([i for i, row in enumerate(rows) if row["cell_id"] == cell]) for cell in cells
    ]
    rng = np.random.default_rng(seed)
    bootstrap_cells = rng.integers(0, len(cells), size=(1000, len(cells)))
    baseline = predictions["train_prevalence"]
    baseline_losses = np.array([np.mean((y[ix] - baseline[ix]) ** 2) for ix in indices])
    scientist_baseline = predictions["scientist_only_nuisance"]
    scientist_losses = np.array([np.mean((y[ix] - scientist_baseline[ix]) ** 2) for ix in indices])
    results = {}
    for method, p in predictions.items():
        metrics = classification_metrics(y, p)
        cell_losses = np.array([np.mean((y[ix] - p[ix]) ** 2) for ix in indices])
        improvement = baseline_losses - cell_losses
        nuisance_improvement = scientist_losses - cell_losses
        aucs = []
        for sample in bootstrap_cells:
            ix = np.concatenate([indices[index] for index in sample])
            if len(np.unique(y[ix])) == 2:
                aucs.append(float(roc_auc_score(y[ix], p[ix])))
        metrics.update(
            {
                "cell_weighted_brier": float(cell_losses.mean()),
                "cell_weighted_brier_gain_vs_prevalence": float(improvement.mean()),
                "cell_weighted_brier_gain_vs_prevalence_ci95": np.quantile(
                    improvement[bootstrap_cells].mean(axis=1), [0.025, 0.975]
                ),
                "auroc_group_bootstrap_ci95": np.quantile(aucs, [0.025, 0.975]) if aucs else None,
                "cell_weighted_brier_gain_vs_scientist_nuisance": float(
                    nuisance_improvement.mean()
                ),
                "cell_weighted_brier_gain_vs_scientist_nuisance_ci95": np.quantile(
                    nuisance_improvement[bootstrap_cells].mean(axis=1), [0.025, 0.975]
                ),
            }
        )
        per_scientist = {}
        for scientist in sorted({row["scientist_model"] for row in rows}):
            ix = np.array(
                [index for index, row in enumerate(rows) if row["scientist_model"] == scientist]
            )
            per_scientist[scientist] = {
                "n": len(ix),
                "positive_count": int(y[ix].sum()),
                **classification_metrics(y[ix], p[ix]),
            }
        metrics["per_scientist"] = per_scientist
        results[method] = metrics
    return {
        "n": len(rows),
        "n_cells": len(cells),
        "n_positive": int(y.sum()),
        "positive_prevalence": float(y.mean()),
        "methods": results,
    }


def restricted_folds(rows: list[dict], folds: list[tuple[np.ndarray, np.ndarray]], keep: list[int]):
    """Preserve original cell-fold assignments when examining smaller cohorts."""
    reverse = {old: new for new, old in enumerate(keep)}
    for fold, (train_ix, test_ix) in enumerate(folds):
        train = np.array([reverse[index] for index in train_ix if index in reverse], dtype=int)
        test = np.array([reverse[index] for index in test_ix if index in reverse], dtype=int)
        if len(test):
            if not len(train):
                raise ValueError("Restricted cohort leaves an empty training partition")
            assert {rows[keep[i]]["cell_id"] for i in train}.isdisjoint(
                {rows[keep[i]]["cell_id"] for i in test}
            )
            yield fold, train, test


def classification_task(
    rows: list[dict],
    folds: list,
    task: str,
    target: np.ndarray,
    keep: list[int],
    condition_key: str | None,
    seed: int,
) -> tuple[dict, list[dict]]:
    cohort = feature_rows([rows[index] for index in keep], condition_key)
    y = target[keep]
    methods = [
        "train_prevalence",
        "scientist_only_nuisance",
        "structured_logistic",
        "structured_extra_trees",
    ]
    if condition_key:
        methods.append("known_score_only_logistic")
    predictions = {method: np.full(len(cohort), np.nan) for method in methods}
    assignments = np.full(len(cohort), -1)
    for fold, train_ix, test_ix in restricted_folds(rows, folds, keep):
        train, test = [cohort[index] for index in train_ix], [cohort[index] for index in test_ix]
        train_y = y[train_ix]
        x_train, x_test = structured_features(train, test)
        predictions["train_prevalence"][test_ix] = train_y.mean()
        predictions["scientist_only_nuisance"][test_ix] = scientist_probabilities(
            train, test, train_y
        )
        predictions["structured_logistic"][test_ix] = fit_classifier(
            "logistic", x_train, x_test, train_y, seed
        )
        predictions["structured_extra_trees"][test_ix] = fit_classifier(
            "extra_trees", x_train, x_test, train_y, seed
        )
        if condition_key:
            parent_train = np.array([[row[condition_key]] for row in train])
            parent_test = np.array([[row[condition_key]] for row in test])
            parent_scaler = StandardScaler()
            parent_train = parent_scaler.fit_transform(parent_train)
            parent_test = parent_scaler.transform(parent_test)
            predictions["known_score_only_logistic"][test_ix] = fit_classifier(
                "logistic",
                parent_train,
                parent_test,
                train_y,
                seed,
            )
        assignments[test_ix] = fold
    assert np.all(assignments >= 0)
    assert all(np.all(np.isfinite(p)) for p in predictions.values())
    metrics = evaluate_classification(cohort, y, predictions, seed)
    records = [
        {
            "task": task,
            "example_id": row["example_id"],
            "cell_id": row["cell_id"],
            "fold": int(assignments[index]),
            "accuracy": row["y"],
            "target": int(y[index]),
            "probabilities": {method: float(p[index]) for method, p in predictions.items()},
        }
        for index, row in enumerate(cohort)
    ]
    return metrics, records


def parent_regression(rows: list[dict], folds: list, seed: int) -> tuple[dict, list[dict]]:
    keep = [index for index, row in enumerate(rows) if row.get("parent_local_accuracy") is not None]
    cohort = feature_rows([rows[index] for index in keep], "parent_local_accuracy")
    y = np.array([row["y"] for row in cohort])
    parent = np.array([row["parent_local_accuracy"] for row in cohort])
    residual = y - parent
    methods = [
        "no_change_parent_local",
        "train_mean_accuracy",
        "train_mean_residual",
        "structured_absolute_ridge",
        "structured_residual_ridge",
        "structured_residual_extra_trees",
        "recipe_residual_ridge",
        "lineage_residual_ridge",
    ]
    predictions = {method: np.full(len(cohort), np.nan) for method in methods}
    assignments = np.full(len(cohort), -1)
    for fold, train_ix, test_ix in restricted_folds(rows, folds, keep):
        train, test = [cohort[index] for index in train_ix], [cohort[index] for index in test_ix]
        x_train, x_test = structured_features(train, test)
        predictions["no_change_parent_local"][test_ix] = parent[test_ix]
        predictions["train_mean_accuracy"][test_ix] = y[train_ix].mean()
        predictions["train_mean_residual"][test_ix] = parent[test_ix] + residual[train_ix].mean()
        predictions["structured_absolute_ridge"][test_ix] = ridge_prediction(
            x_train, x_test, y[train_ix]
        )
        predictions["structured_residual_ridge"][test_ix] = parent[test_ix] + ridge_prediction(
            x_train, x_test, residual[train_ix]
        )
        predictions["structured_residual_extra_trees"][test_ix] = parent[
            test_ix
        ] + ExtraTreesRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        ).fit(x_train, residual[train_ix]).predict(x_test)
        for prefix, text_key in [("recipe", "recipe_text"), ("lineage", "lineage_text")]:
            tx_train, tx_test = text_features(train, test, text_key)
            # Parent-local score is included as a numeric input, while the target
            # is the residual relative to that observed pre-plan score.
            predictions[f"{prefix}_residual_ridge"][test_ix] = parent[test_ix] + ridge_prediction(
                sparse.hstack([tx_train, sparse.csr_matrix(parent[train_ix, None])], format="csr"),
                sparse.hstack([tx_test, sparse.csr_matrix(parent[test_ix, None])], format="csr"),
                residual[train_ix],
            )
        assignments[test_ix] = fold
    predictions = {method: np.clip(p, 0, 1) for method, p in predictions.items()}
    assert np.all(assignments >= 0)
    assert all(np.all(np.isfinite(p)) for p in predictions.values())
    cells = sorted({row["cell_id"] for row in cohort})
    indices = [
        np.array([i for i, row in enumerate(cohort) if row["cell_id"] == cell]) for cell in cells
    ]
    samples = np.random.default_rng(seed).integers(0, len(cells), size=(5000, len(cells)))
    baseline_cell_loss = np.array([np.mean(np.abs(y[ix] - parent[ix])) for ix in indices])
    results = {}
    for method, p in predictions.items():
        metrics = regression_metrics(y, p)
        cell_loss = np.array([np.mean(np.abs(y[ix] - p[ix])) for ix in indices])
        improvement = baseline_cell_loss - cell_loss
        metrics.update(
            {
                "cell_weighted_mae": float(cell_loss.mean()),
                "cell_weighted_mae_gain_vs_no_change": float(improvement.mean()),
                "cell_weighted_mae_gain_vs_no_change_ci95": np.quantile(
                    improvement[samples].mean(axis=1), [0.025, 0.975]
                ),
                "delta_mae": float(np.mean(np.abs((p - parent) - residual))),
            }
        )
        results[method] = metrics
    records = [
        {
            "example_id": row["example_id"],
            "cell_id": row["cell_id"],
            "fold": int(assignments[index]),
            "y": float(y[index]),
            "preplan_parent_local_accuracy": float(parent[index]),
            "target_delta_vs_parent_local": float(residual[index]),
            "predictions": {method: float(p[index]) for method, p in predictions.items()},
        }
        for index, row in enumerate(cohort)
    ]
    return {
        "n": len(cohort),
        "n_cells": len(cells),
        "methods": results,
        "target_delta_vs_parent_local_mean": float(residual.mean()),
        "target_delta_vs_parent_local_std": float(residual.std()),
        "warning": "Official child accuracy minus self-reported local parent accuracy can combine training change with evaluation-protocol mismatch. This is not a clean causal training gain.",
    }, records


def comparator_delta_regression(
    rows: list[dict], folds: list, seed: int
) -> tuple[dict, list[dict]]:
    keep = [
        index
        for index, row in enumerate(rows)
        if row.get("comparator_accuracy") is not None
        and row.get("comparator_official_accuracy") is not None
    ]
    cohort = feature_rows([rows[index] for index in keep], "comparator_accuracy")
    target = np.array([row["y"] - row["comparator_official_accuracy"] for row in cohort])
    known = np.array([row["comparator_accuracy"] for row in cohort])
    methods = [
        "zero_change",
        "train_mean_change",
        "known_comparator_only_ridge",
        "structured_delta_ridge",
        "structured_delta_extra_trees",
        "recipe_delta_ridge",
        "lineage_delta_ridge",
    ]
    predictions = {method: np.full(len(cohort), np.nan) for method in methods}
    assignments = np.full(len(cohort), -1)
    for fold, train_ix, test_ix in restricted_folds(rows, folds, keep):
        train, test = [cohort[index] for index in train_ix], [cohort[index] for index in test_ix]
        x_train, x_test = structured_features(train, test)
        predictions["zero_change"][test_ix] = 0
        predictions["train_mean_change"][test_ix] = target[train_ix].mean()
        known_scaler = StandardScaler()
        predictions["known_comparator_only_ridge"][test_ix] = ridge_prediction(
            known_scaler.fit_transform(known[train_ix, None]),
            known_scaler.transform(known[test_ix, None]),
            target[train_ix],
        )
        predictions["structured_delta_ridge"][test_ix] = ridge_prediction(
            x_train, x_test, target[train_ix]
        )
        predictions["structured_delta_extra_trees"][test_ix] = (
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=-1,
            )
            .fit(x_train, target[train_ix])
            .predict(x_test)
        )
        for prefix, text_key in [("recipe", "recipe_text"), ("lineage", "lineage_text")]:
            tx_train, tx_test = text_features(train, test, text_key)
            predictions[f"{prefix}_delta_ridge"][test_ix] = ridge_prediction(
                sparse.hstack([tx_train, sparse.csr_matrix(known[train_ix, None])], format="csr"),
                sparse.hstack([tx_test, sparse.csr_matrix(known[test_ix, None])], format="csr"),
                target[train_ix],
            )
        assignments[test_ix] = fold
    predictions = {method: np.clip(p, -1, 1) for method, p in predictions.items()}
    assert np.all(assignments >= 0)
    assert all(np.all(np.isfinite(p)) for p in predictions.values())
    cells = sorted({row["cell_id"] for row in cohort})
    indices = [
        np.array([i for i, row in enumerate(cohort) if row["cell_id"] == cell]) for cell in cells
    ]
    samples = np.random.default_rng(seed).integers(0, len(cells), size=(5000, len(cells)))
    baseline_cell_loss = np.array([np.mean(np.abs(target[ix])) for ix in indices])
    results = {}
    for method, p in predictions.items():
        metrics = regression_metrics(target, p)
        cell_loss = np.array([np.mean(np.abs(target[ix] - p[ix])) for ix in indices])
        improvement = baseline_cell_loss - cell_loss
        metrics.update(
            {
                "cell_weighted_mae": float(cell_loss.mean()),
                "cell_weighted_mae_gain_vs_zero_change": float(improvement.mean()),
                "cell_weighted_mae_gain_vs_zero_change_ci95": np.quantile(
                    improvement[samples].mean(axis=1), [0.025, 0.975]
                ),
                "practical_sign_accuracy": float(
                    np.mean(
                        np.where(p > 0.02, 1, np.where(p < -0.02, -1, 0))
                        == np.where(target > 0.02, 1, np.where(target < -0.02, -1, 0))
                    )
                ),
            }
        )
        results[method] = metrics
    records = [
        {
            "example_id": row["example_id"],
            "cell_id": row["cell_id"],
            "fold": int(assignments[index]),
            "target_official_delta": float(target[index]),
            "known_preplan_comparator_accuracy": float(known[index]),
            "comparator_card_id": row["comparator_card_id"],
            "predicted_deltas": {method: float(p[index]) for method, p in predictions.items()},
        }
        for index, row in enumerate(cohort)
    ]
    return {
        "n": len(cohort),
        "n_cells": len(cells),
        "methods": results,
        "n_improve_over_2pp": int(np.sum(target > 0.02)),
        "n_degrade_over_2pp": int(np.sum(target < -0.02)),
        "n_within_2pp": int(np.sum(np.abs(target) <= 0.02)),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "warning": "Official score of the comparator's producing card defines the reference outcome. Matching a card does not guarantee exact intermediate-artifact identity. ±2 pp is a practical threshold, not a statistical significance test.",
    }, records


def render_table(results: dict) -> str:
    lines = []
    for name, task in results["classification"].items():
        lines.extend(
            [
                f"{name}: {task['n']} checkpoints / {task['n_cells']} cells; {task['n_positive']} positive.",
                "",
                "| Method | AUROC | AP | Balanced accuracy | Brier | Cell Brier gain, 95% CI |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, metrics in task["methods"].items():
            lo, hi = metrics["cell_weighted_brier_gain_vs_prevalence_ci95"]
            lines.append(
                f"| {method} | {metrics['auroc']:.3f} | {metrics['average_precision']:.3f} | "
                f"{metrics['balanced_accuracy_at_0_5']:.3f} | {metrics['brier']:.3f} | "
                f"{metrics['cell_weighted_brier_gain_vs_prevalence']:.3f} [{lo:.3f}, {hi:.3f}] |"
            )
        lines.append("")
    regression = results["parent_conditioned_accuracy"]
    lines.extend(
        [
            f"Parent-conditioned official accuracy: {regression['n']} checkpoints / {regression['n_cells']} cells.",
            "",
            "| Method | MAE (pp) | RMSE (pp) | Cell MAE gain vs no-change (pp), 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, metrics in regression["methods"].items():
        lo, hi = metrics["cell_weighted_mae_gain_vs_no_change_ci95"]
        lines.append(
            f"| {method} | {100 * metrics['mae']:.3f} | {100 * metrics['rmse']:.3f} | "
            f"{100 * metrics['cell_weighted_mae_gain_vs_no_change']:.3f} [{100 * lo:.3f}, {100 * hi:.3f}] |"
        )
    delta = results["comparator_official_delta"]
    lines.extend(
        [
            "",
            f"Official change from declared comparator: {delta['n']} checkpoints / {delta['n_cells']} cells.",
            "",
            "| Method | Delta MAE (pp) | Delta RMSE (pp) | Cell MAE gain vs zero change (pp), 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, metrics in delta["methods"].items():
        lo, hi = metrics["cell_weighted_mae_gain_vs_zero_change_ci95"]
        lines.append(
            f"| {method} | {100 * metrics['mae']:.3f} | {100 * metrics['rmse']:.3f} | "
            f"{100 * metrics['cell_weighted_mae_gain_vs_zero_change']:.3f} [{100 * lo:.3f}, {100 * hi:.3f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/analysis/outcome_prediction/decision_tasks")
    )
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, audit = load_examples(args.examples, False)
    folds = split_groups(rows, args.folds, args.seed)
    all_indices = list(range(len(rows)))
    accuracy = np.array([row["y"] for row in rows])
    tasks = [
        ("accuracy_below_0_5", (accuracy < 0.5).astype(int), all_indices, None),
        ("accuracy_at_least_0_7", (accuracy >= 0.7).astype(int), all_indices, None),
    ]
    parent_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("parent_local_accuracy") is not None
        and row.get("parent_accuracy") is not None
        and row.get("parent_accuracy_exact_match", False)
    ]
    parent_improvement = np.array(
        [
            int(row["y"] - row["parent_accuracy"] > 0.02)
            if row.get("parent_accuracy") is not None
            else 0
            for row in rows
        ]
    )
    if len(parent_indices) >= 20 and len(np.unique(parent_improvement[parent_indices])) == 2:
        tasks.append(
            (
                "official_improvement_over_2pp_given_preplan_parent_local",
                parent_improvement,
                parent_indices,
                "parent_local_accuracy",
            )
        )
    comparator_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("comparator_accuracy") is not None
        and row.get("comparator_official_accuracy") is not None
    ]
    comparator_delta = np.array(
        [
            row["y"] - row["comparator_official_accuracy"]
            if row.get("comparator_official_accuracy") is not None
            else 0
            for row in rows
        ]
    )
    tasks.extend(
        [
            (
                "official_gain_over_comparator_2pp",
                (comparator_delta > 0.02).astype(int),
                comparator_indices,
                "comparator_accuracy",
            ),
            (
                "official_loss_under_comparator_2pp",
                (comparator_delta < -0.02).astype(int),
                comparator_indices,
                "comparator_accuracy",
            ),
        ]
    )
    classification, prediction_rows = {}, []
    for name, target, keep, condition_key in tasks:
        print(f"Evaluating {name} ({len(keep)} checkpoints)...", flush=True)
        metrics, records = classification_task(
            rows, folds, name, target, keep, condition_key, args.seed
        )
        classification[name] = metrics
        prediction_rows.extend(records)
    print("Evaluating parent-conditioned accuracy...", flush=True)
    regression, regression_rows = parent_regression(rows, folds, args.seed)
    print("Evaluating official comparator delta...", flush=True)
    delta, delta_rows = comparator_delta_regression(rows, folds, args.seed)
    results = {
        "classification": classification,
        "parent_conditioned_accuracy": regression,
        "comparator_official_delta": delta,
        "audit": audit,
        "protocol": {
            "seed": args.seed,
            "folds": len(folds),
            "splits": "Full eligible-cohort split_groups assignments; parent subsets inherit those assignments.",
            "classification": "Fixed logistic C=1 and ExtraTrees 300 trees depth=4 leaf=4; probability threshold=0.5; no model or threshold tuning.",
            "structured_features": "Train-only imputed/scaled numeric recipe features plus method, parent_kind, peft, precision, scheduler; dataset-path category keys are omitted.",
            "parent_conditioning": "Only parent_local_accuracy documented before target plan time enters parent features. Declared comparator_accuracy is a known input for comparator-conditioned tasks. Current local_accuracy and retrospective official parent/comparator scores never enter any predictor.",
            "improvement_label": "Official child accuracy > official parent accuracy +0.02 for verified exact-parent pairs. Comparator tasks use official child minus comparator-producing-card official accuracy, with ±0.02 practical thresholds. Official reference scores define labels only.",
            "uncertainty": "1000 cell bootstraps for classification and 5000 for residual error; descriptive CIs do not correct OOF training dependence or exploratory multiple comparisons.",
            "scope": "Only eligible checkpoints with official outcomes; low performing means accuracy below 0.5, not execution failure.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", results)
    write_jsonl(args.output_dir / "classification_predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "parent_conditioned_predictions.jsonl", regression_rows)
    write_jsonl(args.output_dir / "comparator_delta_predictions.jsonl", delta_rows)
    write_json(
        args.output_dir / "fold_assignments.json",
        [
            {
                "fold": index,
                "train_cells": sorted({rows[i]["cell_id"] for i in train}),
                "test_cells": sorted({rows[i]["cell_id"] for i in test}),
            }
            for index, (train, test) in enumerate(folds)
        ],
    )
    output = render_table(clean_json(results))
    (args.output_dir / "table.md").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
