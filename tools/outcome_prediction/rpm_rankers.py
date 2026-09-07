"""Retrospective immediate-checkpoint pair ranking with whole-cell nested CV.

This is an adaptation of pair ranking, not a reproduction of a search-tree
subtree-max target. Observed checkpoints in one scientist run were not necessarily
available simultaneously. No current/parent scores, identifiers, free plan prose,
or outcomes are model inputs. The fixed model is recipe+numeric logistic C=1;
all tuned comparisons are exploratory on the existing cohort.

Public interface: make_pairs(rows), fit_ranker(train_rows, spec), and
evaluate_supplied_pairs(rows, pairs). Pairs name a_id and b_id; probabilities
mean P(immediate official checkpoint accuracy of A > B).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from .benchmark import load_examples, split_groups, write_json, write_jsonl
except ImportError:
    from benchmark import load_examples, split_groups, write_json, write_jsonl


SCHEMA = "rpm-immediate-pair-ranking-v1"
TRAIN_GAP = 0.01
SAFE_NUMERIC = {
    "data_components",
    "data_examples",
    "planned_hours",
    "n_direct_parents",
    "lineage_length",
    "ancestor_data_examples_sum",
    "merge_member_count",
    "merge_intermediate_count",
}
SAFE_HP = {
    "batch_size",
    "beta",
    "epochs",
    "grad_accum",
    "lr",
    "max_completion_length",
    "max_prompt_length",
    "max_seq_len",
    "max_steps",
    "num_generations",
    "save_steps",
    "seed",
    "warmup",
    "weight_decay",
}


def numeric_allowed(key: str, view: str) -> bool:
    name = key.removeprefix("log10_")
    if view == "recipe" and name in {"lineage_length", "ancestor_data_examples_sum"}:
        return False
    return (
        name in SAFE_NUMERIC
        or name.removeprefix("hp_") in SAFE_HP
        and name.startswith("hp_")
        or name.startswith("merge_weight_")
        and name.removeprefix("merge_weight_").isdigit()
    )


def canonical_document(row: dict, view: str) -> str:
    """Construct documents from canonical objects, never stored free-text fields."""
    if view == "recipe":
        document = json.dumps(row.get("recipe", {}), sort_keys=True)
    elif view == "lineage":
        document = "\n".join(
            f"step {index + 1}: " + json.dumps(item.get("recipe", {}), sort_keys=True)
            for index, item in enumerate(row.get("lineage", []))
        )
    else:
        raise ValueError(f"Unknown canonical view: {view}")
    return "recipepresent " + document


class FeatureMap:
    def __init__(self, view: str, use_numeric: bool = True):
        self.view = view
        self.use_numeric = use_numeric
        self.text = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=12000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w.]+\b",
        )

    def _numeric(self, rows: list[dict]) -> np.ndarray:
        values = np.array(
            [[r.get("numeric_features", {}).get(k, np.nan) for k in self.keys] for r in rows],
            dtype=float,
        )
        values[~np.isfinite(values)] = np.nan
        return values

    def fit(self, rows: list[dict]) -> FeatureMap:
        self.text.fit([canonical_document(r, self.view) for r in rows])
        self.keys = (
            sorted(
                {
                    k
                    for r in rows
                    for k in r.get("numeric_features", {})
                    if numeric_allowed(k, self.view)
                }
            )
            if self.use_numeric
            else []
        )
        if self.keys:
            self.imputer = SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            )
            x = self.imputer.fit_transform(self._numeric(rows))
            self.scaler = StandardScaler().fit(x)
        return self

    def transform(self, rows: list[dict]):
        text = self.text.transform([canonical_document(r, self.view) for r in rows])
        if not self.keys:
            return text.tocsr()
        numeric = self.scaler.transform(self.imputer.transform(self._numeric(rows)))
        return sparse.hstack([text, sparse.csr_matrix(numeric)], format="csr")


def sibling_key(row: dict) -> tuple[str, ...] | None:
    """Producing weight-parent set; unresolved ancestry is not a strict sibling."""
    if (
        not row.get("lineage_complete", False)
        or "parent_ids" not in row
        or row.get("recipe", {}).get("method") == "merge"
    ):
        return None
    parents = tuple(sorted(set(row["parent_ids"])))
    return parents if len(parents) <= 1 else None


def make_pairs(
    rows: list[dict], min_gap: float = TRAIN_GAP, sibling_only: bool = False
) -> list[dict]:
    if min_gap < 0:
        raise ValueError("min_gap must be nonnegative")
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[str(row["cell_id"])].append(row)
    pairs = []
    for cell, group in sorted(by_cell.items()):
        for a, b in itertools.combinations(sorted(group, key=lambda r: r["example_id"]), 2):
            delta = float(a["y"]) - float(b["y"])
            if abs(delta) + 1e-12 < min_gap or delta == 0:
                continue
            parents = sibling_key(a)
            siblings = parents is not None and parents == sibling_key(b)
            if sibling_only and not siblings:
                continue
            identity = a["example_id"] + "\n" + b["example_id"]
            pairs.append(
                {
                    "pair_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                    "a_id": a["example_id"],
                    "b_id": b["example_id"],
                    "cell_id": cell,
                    "y_a": float(a["y"]),
                    "y_b": float(b["y"]),
                    "delta": delta,
                    "gap": abs(delta),
                    "a_wins": int(delta > 0),
                    "strict_siblings": siblings,
                    "parent_ids": list(parents) if siblings else None,
                }
            )
    return pairs


def pair_arrays(rows: list[dict], pairs: list[dict]):
    ids = {r["example_id"]: i for i, r in enumerate(rows)}
    a = np.array([ids[p["a_id"]] for p in pairs], dtype=int)
    b = np.array([ids[p["b_id"]] for p in pairs], dtype=int)
    return a, b


def equal_cell_weights(groups: list[str]) -> np.ndarray:
    counts = defaultdict(int)
    for group in groups:
        counts[group] += 1
    # Mean weight one retains the intended C/alpha scale without giving large
    # cells quadratic influence simply because they contain more pairs.
    weights = np.array([1 / counts[g] for g in groups])
    return weights / weights.mean()


def pair_features(left, right, contextual: bool = False):
    """Keep signed intervention differences and optionally their absolute context."""
    difference = left - right
    if not contextual:
        return difference.tocsr()
    return sparse.hstack([difference, (left + right) * 0.5], format="csr")


@dataclass
class Ranker:
    feature_map: FeatureMap
    estimator: Any
    kind: str
    scale: float
    training_pairs: int

    def predict_pairs(self, left_rows: list[dict], right_rows: list[dict]) -> np.ndarray:
        if len(left_rows) != len(right_rows):
            raise ValueError("Pair sides have unequal lengths")
        if not left_rows:
            return np.array([], dtype=float)
        left = self.feature_map.transform(left_rows)
        right = self.feature_map.transform(right_rows)
        if self.estimator is None:
            return np.full(len(left_rows), 0.5)
        if self.kind == "pointwise_ridge":
            # Match the existing pointwise baseline's clipped accuracy outputs.
            d = np.clip(self.estimator.predict(left), 0, 1) - np.clip(
                self.estimator.predict(right), 0, 1
            )
            return expit(d / self.scale)
        difference = pair_features(left, right, self.kind == "contextual_forest")
        if self.kind == "ridge":
            return expit(self.estimator.predict(difference) / self.scale)
        forward = self.estimator.predict_proba(difference)[:, 1]
        reversed_features = pair_features(right, left, self.kind == "contextual_forest")
        reverse = self.estimator.predict_proba(reversed_features)[:, 1]
        # Explicit symmetrization is required for the nonlinear classifier;
        # it also makes the interface promise hold to numerical precision.
        return (forward + 1 - reverse) / 2


def fit_ranker(
    train_rows: list[dict],
    spec: dict,
    seed: int = 20260905,
    min_gap: float = TRAIN_GAP,
    train_siblings_only: bool = False,
) -> Ranker:
    fmap = FeatureMap(spec["view"], spec.get("numeric", True)).fit(train_rows)
    x = fmap.transform(train_rows)
    y = np.array([r["y"] for r in train_rows])
    kind = spec["kind"]
    if kind == "pointwise_ridge":
        estimator = Ridge(alpha=spec["alpha"], solver="lsqr").fit(x, y)
        return Ranker(fmap, estimator, kind, max(float(y.std()), 0.01), 0)
    pairs = make_pairs(train_rows, min_gap, sibling_only=train_siblings_only)
    if not pairs:
        return Ranker(fmap, None, kind, 1.0, 0)
    a, b = pair_arrays(train_rows, pairs)
    contextual = kind == "contextual_forest"
    diff = sparse.vstack(
        [pair_features(x[a], x[b], contextual), pair_features(x[b], x[a], contextual)], format="csr"
    )
    delta = y[a] - y[b]
    targets = np.r_[delta, -delta] if kind == "ridge" else np.r_[delta > 0, delta < 0]
    weights = equal_cell_weights([p["cell_id"] for p in pairs])
    weights = np.r_[weights, weights]
    if kind == "logistic":
        estimator = LogisticRegression(
            C=spec["C"], fit_intercept=False, max_iter=3000, solver="lbfgs", random_state=seed
        )
    elif kind == "ridge":
        estimator = Ridge(alpha=spec["alpha"], fit_intercept=False, solver="lsqr")
    elif kind in {"forest", "contextual_forest"}:
        estimator = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"],
            max_features=1.0,
            n_jobs=1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown ranker kind: {kind}")
    estimator.fit(diff, targets, sample_weight=weights)
    return Ranker(fmap, estimator, kind, max(float(delta.std()), 0.01), len(pairs))


def candidate_methods() -> dict[str, list[dict]]:
    result = {
        "fixed_recipe_numeric_logistic_C1": [
            {"view": "recipe", "kind": "logistic", "C": 1.0, "numeric": True}
        ]
    }
    for view in ("recipe", "lineage"):
        result[f"exploratory_{view}_numeric_logistic"] = [
            {"view": view, "kind": "logistic", "C": c, "numeric": True}
            for c in (0.01, 0.1, 1.0, 10.0)
        ]
        result[f"exploratory_{view}_numeric_pairwise_ridge"] = [
            {"view": view, "kind": "ridge", "alpha": a, "numeric": True}
            for a in (0.1, 1.0, 10.0, 100.0)
        ]
        result[f"exploratory_{view}_numeric_forest"] = [
            {
                "view": view,
                "kind": "forest",
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "numeric": True,
            }
            for depth, leaf in ((3, 4), (None, 4), (3, 12), (None, 12))
        ]
        result[f"exploratory_{view}_numeric_contextual_forest"] = [
            {
                "view": view,
                "kind": "contextual_forest",
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "numeric": True,
            }
            for depth, leaf in ((3, 4), (None, 4), (3, 12), (None, 12))
        ]
        result[f"baseline_nested_{view}_pointwise_ridge"] = [
            {"view": view, "kind": "pointwise_ridge", "alpha": a, "numeric": False}
            for a in (0.01, 0.1, 1.0, 10.0, 100.0)
        ]
    return result


def group_folds(rows: list[dict], n_folds: int = 8):
    groups = [str(r["cell_id"]) for r in rows]
    count = min(n_folds, len(set(groups)))
    if count < 2:
        raise ValueError("At least two cells are required")
    return list(GroupKFold(n_splits=count).split(np.arange(len(rows)), groups=groups))


def select_spec(
    train: list[dict], candidates: list[dict], seed: int, train_siblings_only: bool = False
) -> tuple[dict, dict]:
    kind = candidates[0]["kind"]
    objective = (
        "macro_cell_pointwise_mae" if kind == "pointwise_ridge" else "macro_cell_pair_log_loss"
    )
    if train_siblings_only and kind != "pointwise_ridge":
        objective = "macro_cell_strict_sibling_pair_log_loss"
    if len(candidates) == 1:
        return candidates[0], {"fixed": True, "objective": None, "candidates": []}
    folds = group_folds(train, 4)
    losses = [[] for _ in candidates]
    membership = []
    for inner_fold, (tr, va) in enumerate(folds):
        training, validation = [train[i] for i in tr], [train[i] for i in va]
        membership.append(
            {
                "inner_fold": inner_fold,
                "train_cells": sorted({r["cell_id"] for r in training}),
                "validation_cells": sorted({r["cell_id"] for r in validation}),
            }
        )
        pairs = make_pairs(validation, sibling_only=train_siblings_only)
        by_id = {r["example_id"]: r for r in validation}
        for index, candidate in enumerate(candidates):
            fitted = fit_ranker(
                training, candidate, seed + inner_fold, train_siblings_only=train_siblings_only
            )
            if kind == "pointwise_ridge":
                values = np.clip(
                    fitted.estimator.predict(fitted.feature_map.transform(validation)), 0, 1
                )
                errors = np.abs(values - np.array([r["y"] for r in validation]))
                groups = [r["cell_id"] for r in validation]
            else:
                if not pairs:
                    continue
                probabilities = fitted.predict_pairs(
                    [by_id[p["a_id"]] for p in pairs], [by_id[p["b_id"]] for p in pairs]
                )
                truth = np.array([p["a_wins"] for p in pairs])
                probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
                errors = -(truth * np.log(probabilities) + (1 - truth) * np.log1p(-probabilities))
                groups = [p["cell_id"] for p in pairs]
            for cell in sorted(set(groups)):
                mask = np.array([g == cell for g in groups])
                losses[index].append(float(errors[mask].mean()))
    means = [float(np.mean(loss)) if loss else float("inf") for loss in losses]
    winner = int(np.argmin(means))
    return candidates[winner], {
        "fixed": False,
        "objective": objective,
        "inner_folds": membership,
        "candidates": [
            {"spec": spec, "mean_loss": loss, "cell_evaluations": len(values)}
            for spec, loss, values in zip(candidates, means, losses)
        ],
        "selected_index": winner,
    }


def score_pairs(
    records: list[dict], method: str, bootstraps: int = 2000, seed: int = 20260905
) -> dict:
    if not records:
        return {"pairs": 0, "cells": 0}
    by_cell = defaultdict(list)
    for row in records:
        by_cell[row["cell_id"]].append(row)
    summaries = []
    for cell, group in sorted(by_cell.items()):
        p = np.array([r["probabilities"][method] for r in group])
        truth = np.array([r["a_wins"] for r in group])
        gap = np.array([r["gap"] for r in group])
        tie = np.isclose(p, 0.5, atol=1e-12, rtol=0)
        credit = np.where(tie, 0.5, ((p > 0.5) == truth).astype(float))
        summaries.append(
            {
                "cell_id": cell,
                "pairs": len(group),
                "accuracy": float(credit.mean()),
                "regret": float(((1 - credit) * gap).mean()),
                "brier": float(((p - truth) ** 2).mean()),
                "ties": int(tie.sum()),
            }
        )
    n = np.array([s["pairs"] for s in summaries])
    acc = np.array([s["accuracy"] for s in summaries])
    regret = np.array([s["regret"] for s in summaries])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(n), size=(bootstraps, len(n)))

    def interval(values):
        return np.quantile(values, [0.025, 0.975]).tolist() if bootstraps else None

    return {
        "pairs": len(records),
        "cells": len(summaries),
        "macro_cell_accuracy": float(acc.mean()),
        "micro_pair_accuracy": float(np.average(acc, weights=n)),
        "macro_cell_regret": float(regret.mean()),
        "micro_pair_regret": float(np.average(regret, weights=n)),
        "macro_cell_brier": float(np.mean([s["brier"] for s in summaries])),
        "ties": sum(s["ties"] for s in summaries),
        "macro_cell_accuracy_ci95": interval(acc[draws].mean(axis=1)),
        "micro_pair_accuracy_ci95": interval(
            (acc[draws] * n[draws]).sum(axis=1) / n[draws].sum(axis=1)
        ),
        "macro_cell_regret_ci95": interval(regret[draws].mean(axis=1)),
        "micro_pair_regret_ci95": interval(
            (regret[draws] * n[draws]).sum(axis=1) / n[draws].sum(axis=1)
        ),
        "per_cell": summaries,
    }


def evaluate_supplied_pairs(
    rows: list[dict],
    pairs: list[dict],
    *,
    n_folds: int = 8,
    seed: int = 20260905,
    output_dir: Path | None = None,
    methods: dict[str, list[dict]] | None = None,
    outer_assignments: list[dict] | None = None,
    train_siblings_only: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Refit on each outer training partition and score supplied held-out pairs.

    Training pairs use training-cell pairs with gap >= .01; train_siblings_only
    further restricts both training and inner validation to strict siblings.
    Supplied pair selection only controls evaluation. No supplied target/probability is
    read: pair labels are recomputed from the primary eligible examples.
    """
    if any(not r.get("eligible", False) or r.get("y") is None for r in rows):
        raise ValueError("Rows must already be the primary labeled eligible cohort")
    by_id = {r["example_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate example IDs")
    checked = []
    seen = set()
    for supplied in pairs:
        a, b = by_id[supplied["a_id"]], by_id[supplied["b_id"]]
        identity = tuple(sorted((a["example_id"], b["example_id"])))
        if a["cell_id"] != b["cell_id"] or a is b or identity in seen:
            raise ValueError("Pairs must be unique, distinct, and within one cell")
        seen.add(identity)
        label = make_pairs([a, b], 0)
        if not label:
            raise ValueError("Equal-outcome pairs have no binary winner")
        label = label[0]
        # Preserve the supplied A/B order for direct comparison with blind judges.
        if label["a_id"] != supplied["a_id"]:
            label.update(
                a_id=a["example_id"],
                b_id=b["example_id"],
                y_a=a["y"],
                y_b=b["y"],
                delta=a["y"] - b["y"],
                a_wins=int(a["y"] > b["y"]),
            )
        label["pair_id"] = supplied.get("pair_id", label["pair_id"])
        checked.append(label)
    methods = methods or candidate_methods()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if outer_assignments is None:
        folds = split_groups(rows, n_folds, seed)
    else:
        positions = {r["example_id"]: i for i, r in enumerate(rows)}
        folds = []
        all_test = []
        for assignment in outer_assignments:
            train_ids, test_ids = set(assignment["train_ids"]), set(assignment["test_ids"])
            if train_ids & test_ids or train_ids | test_ids != set(positions):
                raise ValueError("Frozen fold does not partition the entire eligible cohort")
            tr = np.array([positions[key] for key in assignment["train_ids"]])
            te = np.array([positions[key] for key in assignment["test_ids"]])
            if {rows[i]["cell_id"] for i in tr} & {rows[i]["cell_id"] for i in te}:
                raise ValueError("Frozen fold splits a cell between training and test")
            folds.append((tr, te))
            all_test.extend(test_ids)
        if len(all_test) != len(rows) or set(all_test) != set(positions):
            raise ValueError("Frozen test folds must cover each eligible example exactly once")
    outputs, assignments = [], []
    for fold, (tr, te) in enumerate(folds):
        train, test = [rows[i] for i in tr], [rows[i] for i in te]
        test_ids = {r["example_id"] for r in test}
        test_pairs = [p for p in checked if p["a_id"] in test_ids]
        if any(p["b_id"] not in test_ids for p in test_pairs):
            raise AssertionError("Pair split across outer folds")
        assignment = {
            "fold": fold,
            "train_cells": sorted({r["cell_id"] for r in train}),
            "test_cells": sorted({r["cell_id"] for r in test}),
            "train_ids": sorted(r["example_id"] for r in train),
            "test_ids": sorted(test_ids),
            "methods": {},
            "train_siblings_only": train_siblings_only,
            "train_pair_count": len(make_pairs(train, sibling_only=train_siblings_only)),
        }
        selected = {}
        for name, candidates in methods.items():
            spec, selection = select_spec(
                train, candidates, seed + fold * 100, train_siblings_only=train_siblings_only
            )
            selected[name] = spec
            assignment["methods"][name] = {"spec": spec, "selection": selection}
        # Persist every choice before applying any model to this held-out fold.
        if output_dir:
            write_json(output_dir / f"fold-{fold:02d}-frozen.json", assignment)
        predictions = {"random_choice": np.full(len(test_pairs), 0.5)}
        for name, spec in selected.items():
            fitted = fit_ranker(
                train, spec, seed + fold * 100, train_siblings_only=train_siblings_only
            )
            predictions[name] = fitted.predict_pairs(
                [by_id[p["a_id"]] for p in test_pairs], [by_id[p["b_id"]] for p in test_pairs]
            )
        for index, pair in enumerate(test_pairs):
            outputs.append(
                {
                    **pair,
                    "fold": fold,
                    "probabilities": {m: float(p[index]) for m, p in predictions.items()},
                }
            )
        assignments.append(assignment)
        print(
            f"Outer cell fold {fold + 1}/{len(folds)}: {len(test)} checkpoints, "
            f"{len(test_pairs)} held-out pairs",
            flush=True,
        )
    if len(outputs) != len(checked):
        raise AssertionError("Some supplied pairs did not receive an outer-fold prediction")
    return outputs, assignments


def write_report(output_dir: Path, results: dict) -> None:
    method_names = {
        method for evaluation in results["evaluations"].values() for method in evaluation
    }
    probability_note = (
        "Ridge outputs use a monotonic sigmoid link with a training-only scale; "
        "those probabilities are uncalibrated ranking scores. "
        if any("ridge" in name for name in method_names)
        else ""
    )
    context_note = (
        " Contextual forests receive both the signed feature difference and "
        "the pair mean. Swapping candidates negates the difference and preserves "
        "the mean, so absolute learning-rate/data levels can affect the ranking. "
        "This model extension was added after inspecting earlier learned-model results."
        if any("contextual_forest" in name for name in method_names)
        else ""
    )
    fixed_note = (
        "The recipe+numeric logistic C=1 model is fixed. "
        if "fixed_recipe_numeric_logistic_C1" in method_names
        else "All reported learned models use inner-fold hyperparameter selection. "
    )
    training_scope = (
        "Training and inner selection use only complete-lineage, non-merge sibling "
        "pairs sharing the exact weight-parent set (at most one parent). This is an "
        "exploratory follow-up sensitivity added after inspecting the original results."
        if results.get("train_siblings_only")
        else "Training and inner selection use all same-cell pairs, including nonsiblings."
    )
    lines = [
        "# Immediate-checkpoint pair ranking",
        "",
        (
            "Retrospective pairs of executed checkpoints; availability as simultaneous "
            "candidates is not established. Outcomes are immediate official accuracies, "
            "not maximum descendant/subtree rewards."
        ),
        "",
        "Eight frozen outer partitions hold out whole cells and match the blind judge's "
        "training-history banks. Four inner whole-cell GroupKFold "
        "folds select exploratory hyperparameters using training labels only. "
        + fixed_note
        + "All training pair sets "
        "use gap ≥ 0.01, with equal total loss weight per training cell. " + training_scope,
        "",
        (
            "Accuracy gives 0.5 credit to a tied model score. Regret is the immediate "
            "accuracy lost by selecting the lower-accuracy member (half the gap on ties), "
            "shown in percentage points. CIs resample entire held-out cells and remain "
            "descriptive given overlapping CV training sets and exploratory comparisons."
        ),
        "",
        probability_note + "Forest predictions "
        "are explicitly antisymmetrized. No score or scientist/cell identifier is a feature."
        + context_note,
        "",
    ]
    for subset, metrics in results["evaluations"].items():
        lines.extend(
            [
                f"## {subset}",
                "",
                "| Method | Pairs / cells | Macro accuracy (95% CI) | Micro accuracy | Macro regret pp |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method, score in metrics.items():
            if not score["pairs"]:
                continue
            lo, hi = score["macro_cell_accuracy_ci95"] or [float("nan")] * 2
            lines.append(
                f"| {method} | {score['pairs']} / {score['cells']} | "
                f"{score['macro_cell_accuracy']:.3f} ({lo:.3f}, {hi:.3f}) | "
                f"{score['micro_pair_accuracy']:.3f} | {score['macro_cell_regret'] * 100:.2f} |"
            )
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", type=Path, default=Path("data/analysis/outcome_prediction/examples.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/rpm/learned"))
    parser.add_argument("--pairs", type=Path, help="Optional supplied JSONL pair IDs/orientations")
    parser.add_argument(
        "--folds-json", type=Path, help="Frozen outer train_ids/test_ids shared with judges"
    )
    parser.add_argument(
        "--train-siblings-only",
        action="store_true",
        help="Exploratory sensitivity: fit/tune pairwise models on strict siblings only",
    )
    parser.add_argument("--methods", help="Comma-separated method names from candidate_methods()")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    rows, audit = load_examples(args.examples, False)
    methods = candidate_methods()
    if args.methods:
        names = args.methods.split(",")
        unknown = sorted(set(names) - set(methods))
        if unknown or len(names) != len(set(names)):
            parser.error(f"Unknown or duplicate method names: {unknown or names}")
        methods = {name: methods[name] for name in names}
    if args.train_siblings_only and any(
        specs[0]["kind"] == "pointwise_ridge" for specs in methods.values()
    ):
        parser.error("--train-siblings-only requires --methods selecting pairwise models only")
    pairs = (
        [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
        if args.pairs
        else make_pairs(rows)
    )
    outputs, folds = evaluate_supplied_pairs(
        rows,
        pairs,
        n_folds=args.folds,
        seed=args.seed,
        output_dir=args.output_dir,
        outer_assignments=json.loads(args.folds_json.read_text()) if args.folds_json else None,
        methods=methods,
        train_siblings_only=args.train_siblings_only,
    )
    write_jsonl(args.output_dir / "pair_predictions.jsonl", outputs)
    write_json(args.output_dir / "fold_assignments.json", folds)
    methods = list(outputs[0]["probabilities"]) if outputs else []
    evaluations = {}
    for siblings in (False, True):
        for gap in (0.01, 0.02):
            subset = [
                p
                for p in outputs
                if p["gap"] + 1e-12 >= gap and (not siblings or p["strict_siblings"])
            ]
            name = ("strict_same_weight_parent" if siblings else "same_cell") + f"_gap_{gap:.2f}"
            evaluations[name] = {
                m: score_pairs(subset, m, args.bootstraps, args.seed) for m in methods
            }
    results = {
        "schema": SCHEMA,
        "cohort": {
            **audit,
            "eligible_checkpoints": len(rows),
            "cells": len({r["cell_id"] for r in rows}),
            "examples_sha256": hashlib.sha256(args.examples.read_bytes()).hexdigest(),
        },
        "outer_split": "frozen_cell_groups" if args.folds_json else "benchmark.split_groups",
        "outer_folds": len(folds),
        "inner_split": "GroupKFold",
        "inner_folds": 4,
        "train_pair_gap": TRAIN_GAP,
        "seed": args.seed,
        "bootstraps": args.bootstraps,
        "train_siblings_only": args.train_siblings_only,
        "exploratory_followup_sensitivity": args.train_siblings_only,
        "target": "immediate_checkpoint_accuracy_difference",
        "simultaneous_candidate_availability_established": False,
        "evaluations": evaluations,
    }
    write_json(args.output_dir / "metrics.json", results)
    write_report(args.output_dir, results)


if __name__ == "__main__":
    main()
