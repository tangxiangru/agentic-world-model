"""Delta world model: predict child accuracy minus parent accuracy.

Only experiments whose parent is a scored checkpoint are used (delta is defined).
Run-held-out 8-fold; metrics are about the change itself.
Usage: python -m tools.wm_study.wm_delta
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

import tools.wm_study.wm as W
from tools.wm_study.wm import load_cards, make_folds, run_weights, structured_features
from tools.wm_study.wm_state import build_states, features as state_features


def parent_acc(c, by_id):
    vals = [by_id[f"{c['cell_id']}/{p}"]["official_accuracy"] for p in c["parents"] if f"{c['cell_id']}/{p}" in by_id and by_id[f"{c['cell_id']}/{p}"]["official_accuracy"] is not None]
    return float(np.mean(vals)) if vals else None


def hgb(seed=0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=seed)


def make_X(rows, kind, memo, keys=None):
    feats = []
    for r in rows:
        f = {}
        if kind in ("flat", "flat+pacc", "all"):
            f.update(structured_features(r))
        if kind in ("flat+pacc", "all", "state"):
            f["parent_acc"] = r["_pacc"]
        if kind in ("state", "all"):
            m = memo[r["example_id"]]
            sf = state_features(r, m["parent_state"], m["sampler_acc"], use_parent_acc=True)
            f.update({("s_" + k if not k.startswith(("s_", "i_", "t_")) else k): v for k, v in sf.items()})
        feats.append(f)
    if keys is None:
        keys = sorted(set().union(*feats))
    X = np.array([[float(f[k]) if (k in f and f[k] is not None) else np.nan for k in keys] for f in feats])
    return X, keys


def metrics(y_delta, p_delta):
    y, p = np.asarray(y_delta), np.asarray(p_delta)
    out = {
        "mae": float(np.abs(y - p).mean()),
        "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "spearman": float(spearmanr(y, p).correlation) if np.std(p) > 0 else float("nan"),
        "sign_acc": float(np.mean((p > 0) == (y > 0))[()]) if True else None,
    }
    big = np.abs(y) >= 0.01
    out["sign_acc_1pt"] = float(np.mean((p[big] > 0) == (y[big] > 0))) if big.any() else float("nan")
    drop = y <= -0.10
    gain = y >= 0.05
    out["auc_drop10"] = float(roc_auc_score(drop, -p)) if 0 < drop.sum() < len(y) and np.std(p) > 0 else float("nan")
    out["auc_gain5"] = float(roc_auc_score(gain, p)) if 0 < gain.sum() < len(y) and np.std(p) > 0 else float("nan")
    out["n_drop10"], out["n_gain5"] = int(drop.sum()), int(gain.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k", "aime2025"])
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/wm_delta"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
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
                continue
            r["_pacc"] = pa
            r["_delta"] = r["official_accuracy"] - pa
            rows.append(r)
        fold_of = make_folds(rows, 8)
        y = np.array([r["_delta"] for r in rows])
        print(f"\n== {bench}: {len(rows)} experiments with a scored parent, {len(fold_of)} sessions. true delta: mean {y.mean():+.3f}, sd {y.std():.3f}, drops>=10pt {int((y<=-0.10).sum())}, gains>=5pt {int((y>=0.05).sum())}")
        print("| predictor of the change | MAE | R² | Spearman | sign acc (|Δ|≥1pt) | AUC: catch ≥10-pt drops | AUC: catch ≥5-pt gains |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        preds = {}
        # baselines
        preds["no change (Δ=0)"] = np.zeros(len(rows))
        mean_delta = np.zeros(len(rows))
        for k in range(8):
            tr = [i for i, r in enumerate(rows) if fold_of[r["cell_id"]] != k]
            te = [i for i, r in enumerate(rows) if fold_of[r["cell_id"]] == k]
            mean_delta[te] = y[tr].mean()
        preds["training-mean change"] = mean_delta
        # regress-to-the-mean baseline: linear in parent accuracy only
        lin = np.zeros(len(rows))
        for k in range(8):
            tr = [i for i, r in enumerate(rows) if fold_of[r["cell_id"]] != k]
            te = [i for i, r in enumerate(rows) if fold_of[r["cell_id"]] == k]
            pa = np.array([rows[i]["_pacc"] for i in tr])
            a, b = np.polyfit(pa, y[tr], 1)
            lin[te] = a * np.array([rows[i]["_pacc"] for i in te]) + b
        preds["linear in parent accuracy only"] = lin
        for name, kind in (("recipe features → Δ", "flat"), ("recipe + parent acc → Δ", "flat+pacc"), ("state + interactions + parent acc → Δ", "state"), ("recipe + state + parent acc → Δ", "all")):
            p = np.zeros(len(rows))
            for k in range(8):
                tr = [r for r in rows if fold_of[r["cell_id"]] != k]
                te_idx = [i for i, r in enumerate(rows) if fold_of[r["cell_id"]] == k]
                te = [rows[i] for i in te_idx]
                Xtr, keys = make_X(tr, kind, memo)
                Xte, _ = make_X(te, kind, memo, keys)
                nan = np.isnan(Xtr).all(0)
                Xtr[:, nan] = 0
                Xte[:, nan] = 0
                ps = np.mean([hgb(s).fit(Xtr, np.array([r["_delta"] for r in tr]), sample_weight=run_weights(tr)).predict(Xte) for s in range(3)], axis=0)
                p[te_idx] = ps
            preds[name] = p
        preds["ensemble: linear-in-parent + recipe+parent Δ"] = (preds["linear in parent accuracy only"] + preds["recipe + parent acc → Δ"]) / 2
        results = {}
        for name, p in preds.items():
            m = metrics(y, p)
            results[name] = m
            print(f"| {name} | {m['mae']:.3f} | {m['r2']:.2f} | {m['spearman']:.2f} | {m['sign_acc_1pt']:.2f} | {m['auc_drop10']:.2f} | {m['auc_gain5']:.2f} |")
        (args.out / f"{bench}.json").write_text(json.dumps({"n": len(rows), "results": results, "predictions": {r["example_id"]: {"delta": float(y[i]), **{k: float(v[i]) for k, v in preds.items()}} for i, r in enumerate(rows)}}, indent=1))


if __name__ == "__main__":
    main()
