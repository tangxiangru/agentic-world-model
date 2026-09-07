"""One table: predict each experiment's official accuracy; compare predictors.

Rows are all labeled experiments with a known parent accuracy (base model's accuracy
for first-round experiments). Predictors: parent accuracy unchanged, the delta world
model (recipe + parent accuracy -> change, run-held-out), the agent, and combinations.
Reports MAE and R^2 for the accuracy and for the change from the parent.
Usage: python -m tools.wm_study.compare_predictors --benchmarks gsm8k aime2025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.wm_study.wm import load_cards, make_folds, run_weights, TreeStructured
from tools.wm_study.wm_delta import make_X, parent_acc
from tools.wm_study.wm_state import build_states

BASE = {"gsm8k": 0.045, "aime2025": 0.05}


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=0)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    ap.add_argument("--agent", type=Path, default=Path("data/analysis/wm_study/agent_predict"))
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k", "aime2025"])
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/compare_predictors.md"))
    ap.add_argument("--continuation-only", action="store_true", help="keep only experiments whose parent is a scored checkpoint")
    args = ap.parse_args()
    lines = []
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
                if r["parents"] or args.continuation_only:
                    continue
                pa = BASE[bench]
            r["_pacc"] = pa
            rows.append(r)
        fold_of = make_folds(rows, 8)
        y = np.array([r["official_accuracy"] for r in rows])
        pa = np.array([r["_pacc"] for r in rows])
        from_base = np.array([not r["parents"] for r in rows])
        preds = {"parent accuracy, unchanged": pa.copy()}
        # recipe-only absolute model (no parent accuracy)
        p = np.zeros(len(y))
        for k in range(8):
            tr = [r for r in rows if fold_of[r["cell_id"]] != k]
            te = [i for i, r in enumerate(rows) if fold_of[r["cell_id"]] == k]
            m = TreeStructured("h", "hgb", False).fit(tr, np.array([r["official_accuracy"] for r in tr]))
            p[te] = m.predict([rows[i] for i in te])
        preds["recipe-only world model (absolute)"] = p
        # delta world model: recipe + parent acc -> change
        X, keys = make_X(rows, "flat+pacc", memo)
        p = np.zeros(len(y))
        for k in range(8):
            tr = [i for i, r in enumerate(rows) if fold_of[r["cell_id"]] != k]
            te = [i for i, r in enumerate(rows) if fold_of[r["cell_id"]] == k]
            Xtr, Xte = X[tr].copy(), X[te].copy()
            nan = np.isnan(Xtr).all(0)
            Xtr[:, nan] = 0
            Xte[:, nan] = 0
            d = np.mean([HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=s).fit(Xtr, (y - pa)[tr], sample_weight=run_weights([rows[i] for i in tr])).predict(Xte) for s in range(3)], axis=0)
            p[te] = np.clip(pa[te] + d, 0, 1)
        preds["delta world model (recipe + parent acc → change)"] = p
        # agent
        ag = np.full(len(y), np.nan)
        cost = 0.0
        for i, r in enumerate(rows):
            f = args.agent / bench / (r["example_id"].replace("/", "__") + ".json")
            if f.exists():
                rec = json.loads(f.read_text())
                v = (rec.get("pred") or {}).get("predicted_accuracy")
                if isinstance(v, (int, float)):
                    ag[i] = float(np.clip(v, 0, 1))
                cost += rec["meta"].get("cost_usd") or 0
        have = ~np.isnan(ag)
        preds["agent (Opus 5, recipe + parent acc + bank)"] = ag
        preds["average of agent and delta world model"] = (ag + preds["delta world model (recipe + parent acc → change)"]) / 2
        lines.append(f"\n## {bench}: {len(rows)} experiments ({int(from_base.sum())} from the base model, {int((~from_base).sum())} continuations); agent answered {int(have.sum())}, cost ${cost:.0f}\n")
        lines.append("| predictor | MAE accuracy | R² accuracy | MAE change | R² change | Spearman change | MAE, continuations only | MAE, from-base only |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, p in preds.items():
            ok = ~np.isnan(p)
            yy, pp, pq = y[ok], p[ok], pa[ok]
            dy, dp = yy - pq, pp - pq
            sp = spearmanr(dy, dp).correlation if np.std(dp) > 0 else float("nan")
            fb = from_base[ok]
            lines.append(f"| {name} | {np.abs(yy - pp).mean():.3f} | {r2(yy, pp):.2f} | {np.abs(dy - dp).mean():.3f} | {r2(dy, dp):.2f} | {sp:.2f} | {np.abs(yy - pp)[~fb].mean():.3f} | {np.abs(yy - pp)[fb].mean():.3f} |")
        lines.append(f"\nAccuracy sd {y.std():.3f}; change sd {(y - pa).std():.3f}; change sd among continuations {(y - pa)[~from_base].std():.3f}.")
    text = "\n".join(lines)
    print(text)
    args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
