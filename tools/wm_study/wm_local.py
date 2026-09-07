"""Local (retrieval-based) delta models: fit on the past experiments most similar to the query.

For each held-out experiment, find its nearest training experiments (standardized recipe
features, plus optional text similarity), then predict its change from the parent with a
model fitted on that subset only. Compared with the global gradient-boosting delta model.
Usage: python -m tools.wm_study.wm_local
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from tools.wm_study.wm import load_cards, make_folds, recipe_text, run_weights
from tools.wm_study.wm_delta import make_X, parent_acc
from tools.wm_study.wm_state import build_states

BASE = {"gsm8k": 0.045, "aime2025": 0.05}
STRONG = ("t_epochs", "t_log_steps", "t_log_n_examples", "t_log_lr", "t_max_seq_len", "parent_acc", "depth", "t_fam_sft", "t_fam_rft", "t_fam_merge", "t_fam_decoding", "t_src_derived", "t_src_self", "t_planned_h")


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k", "aime2025"])
    args = ap.parse_args()
    for bench in args.benchmarks:
        all_rows = load_cards(args.cards, bench, labeled_only=False)
        by_id = {r["example_id"]: r for r in all_rows}
        memo = build_states(by_id, bench)
        rows = []
        for r in all_rows:
            if r["official_accuracy"] is None or not r["lineage_complete"]:
                continue
            pa = parent_acc(r, by_id)
            if pa is None:
                if r["parents"]:
                    continue
                pa = BASE[bench]
            r["_pacc"] = pa
            rows.append(r)
        fold_of = make_folds(rows, 8)
        y = np.array([r["official_accuracy"] for r in rows])
        pa = np.array([r["_pacc"] for r in rows])
        d = y - pa
        from_base = np.array([not r["parents"] for r in rows])
        X, keys = make_X(rows, "flat+pacc", memo)
        med = np.nanmedian(X, 0)
        med = np.where(np.isnan(med), 0, med)
        Xi = np.where(np.isnan(X), med, X)
        sidx = [keys.index(k) for k in STRONG if k in keys]
        texts = [recipe_text(r) for r in rows]
        preds = {}

        def per_fold(fn):
            p = np.zeros(len(y))
            for k in range(8):
                tr = np.array([i for i, r in enumerate(rows) if fold_of[r["cell_id"]] != k])
                te = np.array([i for i, r in enumerate(rows) if fold_of[r["cell_id"]] == k])
                p[te] = fn(tr, te)
            return p

        # global reference
        def global_hgb(tr, te):
            Xtr, Xte = X[tr].copy(), X[te].copy()
            nan = np.isnan(Xtr).all(0)
            Xtr[:, nan] = 0
            Xte[:, nan] = 0
            return np.mean([HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=s).fit(Xtr, d[tr], sample_weight=run_weights([rows[i] for i in tr])).predict(Xte) for s in range(3)], axis=0)
        preds["global gradient boosting (reference)"] = per_fold(global_hgb)

        # standardized feature distance on the strong features + same-regime constraint
        def neighbours(tr, te, k, use_text=False):
            mu, sd = Xi[tr][:, sidx].mean(0), Xi[tr][:, sidx].std(0) + 1e-6
            A = (Xi[tr][:, sidx] - mu) / sd
            B = (Xi[te][:, sidx] - mu) / sd
            dist = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
            if use_text:
                vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b[\w][\w.\-=]*\b")
                T = vec.fit_transform([texts[i] for i in tr])
                Q = vec.transform([texts[i] for i in te])
                sim = (Q @ T.T).toarray()
                dist = dist / (dist.mean() + 1e-9) + (1 - sim)
            # same regime: from-base queries only look at from-base rows and vice versa
            same = from_base[te][:, None] == from_base[tr][None, :]
            dist = np.where(same, dist, np.inf)
            order = np.argsort(dist, axis=1)[:, :k]
            return order, np.take_along_axis(dist, order, axis=1)

        def local_mean(k, use_text=False):
            def fn(tr, te):
                order, dist = neighbours(tr, te, k, use_text)
                out = np.zeros(len(te))
                for i in range(len(te)):
                    idx = tr[order[i]]
                    w = np.exp(-(dist[i] / (np.median(dist[i]) + 1e-9)) ** 2)
                    out[i] = np.average(d[idx], weights=w)
                return out
            return fn

        def local_ridge(k, alpha, use_text=False):
            def fn(tr, te):
                order, dist = neighbours(tr, te, k, use_text)
                out = np.zeros(len(te))
                mu, sd = Xi[tr][:, sidx].mean(0), Xi[tr][:, sidx].std(0) + 1e-6
                for i in range(len(te)):
                    idx = tr[order[i]]
                    w = np.exp(-(dist[i] / (np.median(dist[i]) + 1e-9)) ** 2)
                    A = (Xi[idx][:, sidx] - mu) / sd
                    m = Ridge(alpha=alpha).fit(A, d[idx], sample_weight=w)
                    out[i] = m.predict(((Xi[te[i]][sidx] - mu) / sd)[None, :])[0]
                return out
            return fn

        def local_hgb(k, use_text=False):
            def fn(tr, te):
                order, dist = neighbours(tr, te, k, use_text)
                out = np.zeros(len(te))
                for i in range(len(te)):
                    idx = tr[order[i]]
                    Xtr = X[idx].copy()
                    nan = np.isnan(Xtr).all(0)
                    Xtr[:, nan] = 0
                    Xq = X[te[i]].copy()[None, :]
                    Xq[:, nan] = 0
                    m = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=4, min_samples_leaf=4, l2_regularization=2.0, random_state=0).fit(Xtr, d[idx])
                    out[i] = m.predict(Xq)[0]
                return out
            return fn

        preds["local kernel mean, 15 nearest"] = per_fold(local_mean(15))
        preds["local kernel mean, 30 nearest"] = per_fold(local_mean(30))
        preds["local kernel mean, 15 nearest, features + text"] = per_fold(local_mean(15, True))
        preds["local ridge, 30 nearest"] = per_fold(local_ridge(30, 10.0))
        preds["local ridge, 60 nearest"] = per_fold(local_ridge(60, 10.0))
        preds["local gradient boosting, 40 nearest"] = per_fold(local_hgb(40))
        preds["average: global boosting + local kernel mean 15"] = (preds["global gradient boosting (reference)"] + preds["local kernel mean, 15 nearest"]) / 2

        print(f"\n== {bench}: {len(rows)} experiments ({from_base.sum()} from base, {(~from_base).sum()} continuations)")
        print("| predictor | MAE accuracy | R² accuracy | MAE continuations | MAE first-round | Spearman of change (continuations) |")
        print("|---|---:|---:|---:|---:|---:|")
        print(f"| parent unchanged | {np.abs(d).mean():.3f} | {r2(y, pa):.2f} | {np.abs(d[~from_base]).mean():.3f} | {np.abs(d[from_base]).mean():.3f} | — |")
        for name, pd_ in preds.items():
            p = np.clip(pa + pd_, 0, 1)
            sp = spearmanr(d[~from_base], pd_[~from_base]).correlation
            print(f"| {name} | {np.abs(y - p).mean():.3f} | {r2(y, p):.2f} | {np.abs(y - p)[~from_base].mean():.3f} | {np.abs(y - p)[from_base].mean():.3f} | {sp:.2f} |")


if __name__ == "__main__":
    main()
