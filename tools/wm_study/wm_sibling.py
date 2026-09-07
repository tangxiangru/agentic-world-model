"""Sibling-focused world models: can within-set context improve selection?

Variants (all HGB, run-held-out 8-fold):
  pointwise            current flat recipe regressor
  set-relative         pointwise + each numeric feature's deviation from its sibling-set mean
  pairwise (all-run)   classifier on [a-b, (a+b)/2] over all within-run pairs, scored by mean win prob
  pairwise (sibling)   same, trained on same-parent pairs only
  ensemble             rank-average of pointwise and pairwise (all-run)
Usage: python -m tools.wm_study.wm_sibling
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from tools.wm_study.wm import load_cards, make_folds, point_metrics, run_weights, selection_metrics, structured_features

REL_KEYS = ("t_log_steps", "t_log_n_examples", "t_epochs", "t_log_lr", "t_max_seq_len", "t_planned_h", "t_eff_batch", "t_warmup", "t_script_chars", "t_n_sources", "t_max_mix", "t_temperature", "t_max_tokens", "t_weight_decay")


def set_key(r):
    return (r["cell_id"], tuple(r["parents"]))


def matrix(rows, keys):
    feats = [structured_features(r) for r in rows]
    return np.array([[float(f.get(k, np.nan)) if f.get(k) is not None else np.nan for k in keys] for f in feats])


def relative_block(rows, X, keys):
    """Deviation of selected numeric features from the sibling-set mean (0 if alone)."""
    idx = {k: i for i, k in enumerate(keys)}
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[set_key(r)].append(i)
    R = np.zeros((len(rows), len(REL_KEYS)))
    S = np.zeros((len(rows), 1))
    for members in groups.values():
        S[members, 0] = len(members)
        if len(members) < 2:
            continue
        for j, k in enumerate(REL_KEYS):
            col = X[members, idx[k]]
            if np.all(np.isnan(col)):
                continue
            mu = np.nanmean(col)
            R[members, j] = np.where(np.isnan(col), 0.0, col - mu)
    return np.hstack([R, S])


def hgb_reg():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=0)


def hgb_clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=0)


class Pointwise:
    def __init__(self, relative=False):
        self.relative = relative

    def fit(self, rows, keys):
        X = matrix(rows, keys)
        if self.relative:
            X = np.hstack([X, relative_block(rows, X, keys)])
        self.nan = np.isnan(X).all(0)
        X[:, self.nan] = 0
        self.m = hgb_reg().fit(X, np.array([r["official_accuracy"] for r in rows]), sample_weight=run_weights(rows))
        return self

    def predict(self, rows, keys):
        X = matrix(rows, keys)
        if self.relative:
            X = np.hstack([X, relative_block(rows, X, keys)])
        X[:, self.nan] = 0
        return self.m.predict(X)


class Pairwise:
    def __init__(self, scope="run"):
        self.scope = scope

    def fit(self, rows, keys):
        X = matrix(rows, keys)
        self.nan = np.isnan(X).all(0)
        X[:, self.nan] = 0
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[r["cell_id"] if self.scope == "run" else set_key(r)].append(i)
        F, y, w = [], [], []
        for members in groups.values():
            for i in members:
                for j in members:
                    if i == j or abs(rows[i]["official_accuracy"] - rows[j]["official_accuracy"]) < 0.01:
                        continue
                    F.append(np.concatenate([X[i] - X[j], (X[i] + X[j]) / 2]))
                    y.append(int(rows[i]["official_accuracy"] > rows[j]["official_accuracy"]))
                    w.append(1.0 / (len(members) * (len(members) - 1)))
        self.m = hgb_clf().fit(np.array(F), np.array(y), sample_weight=np.array(w))
        return self

    def predict(self, rows, keys):
        X = matrix(rows, keys)
        X[:, self.nan] = 0
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[set_key(r)].append(i)
        out = np.full(len(rows), 0.5)
        for members in groups.values():
            if len(members) < 2:
                continue
            for i in members:
                others = [j for j in members if j != i]
                A = np.repeat(X[i:i + 1], len(others), 0)
                B = X[others]
                p = (self.m.predict_proba(np.hstack([A - B, (A + B) / 2]))[:, 1] + 1 - self.m.predict_proba(np.hstack([B - A, (A + B) / 2]))[:, 1]) / 2
                out[i] = p.mean()
        return out


def rank_within_sets(rows, score):
    """Convert scores to within-set ranks in [0,1] so different models can be averaged."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[set_key(r)].append(i)
    out = np.zeros(len(rows))
    for members in groups.values():
        if len(members) == 1:
            out[members[0]] = 0.5
            continue
        order = np.argsort(np.argsort([score[i] for i in members]))
        for m, o in zip(members, order):
            out[m] = o / (len(members) - 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k", "aime2025"])
    args = ap.parse_args()
    for bench in args.benchmarks:
        rows = [r for r in load_cards(args.cards, bench) if r["lineage_complete"]]
        keys = sorted(structured_features(rows[0]))
        fold_of = make_folds(rows, 8)
        y = {r["example_id"]: r["official_accuracy"] for r in rows}
        variants = {"pointwise (current)": lambda: Pointwise(False), "set-relative pointwise": lambda: Pointwise(True), "pairwise, all within-run pairs": lambda: Pairwise("run"), "pairwise, sibling pairs only": lambda: Pairwise("set")}
        preds = {}
        for name, make in variants.items():
            pred = {}
            for k in range(8):
                tr = [r for r in rows if fold_of[r["cell_id"]] != k]
                te = [r for r in rows if fold_of[r["cell_id"]] == k]
                m = make().fit(tr, keys)
                for r, p in zip(te, m.predict(te, keys)):
                    pred[r["example_id"]] = float(p)
            preds[name] = pred
        # ensembles by within-set rank averaging
        for a, b, name in (("pointwise (current)", "pairwise, all within-run pairs", "ensemble: pointwise + pairwise(run)"), ("set-relative pointwise", "pairwise, all within-run pairs", "ensemble: set-relative + pairwise(run)"), ("pointwise (current)", "set-relative pointwise", "ensemble: pointwise + set-relative")):
            ra = rank_within_sets(rows, [preds[a][r["example_id"]] for r in rows])
            rb = rank_within_sets(rows, [preds[b][r["example_id"]] for r in rows])
            preds[name] = {r["example_id"]: float((ra[i] + rb[i]) / 2 + 1e-6 * preds[a][r["example_id"]]) for i, r in enumerate(rows)}
        print(f"\n== {bench}: {len(rows)} labeled checkpoints")
        print("| model | sibling pairwise | top-1 | regret | chosen acc | pairwise (5pt sets) | regret (5pt sets) | Spearman (all) |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, pred in preds.items():
            sm = selection_metrics(rows, pred)
            s5 = selection_metrics(rows, pred, min_gap=0.05)
            pm = point_metrics([y[i] for i in pred], [pred[i] for i in pred]) if "pairwise" not in name and "ensemble" not in name else {"spearman": float("nan")}
            print(f"| {name} | {sm['pairwise_acc']:.3f} | {sm['top1_acc']:.3f} | {sm['regret']:.3f} | {sm['chosen_acc']:.3f} | {s5['pairwise_acc']:.3f} | {s5['regret']:.3f} | {pm['spearman']:.2f} |")
        s = selection_metrics(rows, {r["example_id"]: 0.0 for r in rows})
        print(f"| random | 0.500 | | {s['oracle_acc'] - s['random_acc']:.3f} | {s['random_acc']:.3f} | 0.500 | | |\n| oracle | 1.000 | 1.000 | 0 | {s['oracle_acc']:.3f} | 1.000 | 0 | |")


if __name__ == "__main__":
    main()
