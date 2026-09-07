"""Fit the world model per fold and write the per-decision-set WM tool files.

For each benchmark and fold: fit the chosen variant on the fold's training runs
only, estimate its reliability by inner run-held-out CV on those same runs, then
for each decision set of the held-out fold write `predictions.json` (prediction
per candidate + nearest training recipes with their official scores) and a
README explaining how to use it.

Usage: python -m tools.wm_study.wm_tool --variant nested
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from tools.wm_study.decision_sets import short_recipe
from tools.wm_study.wm import decision_sets, load_cards, make_folds, make_variants, point_metrics, recipe_text, selection_metrics

README = """# World-model predictions for this decision set

`predictions.json` gives, per candidate letter:

- `predicted_official_accuracy`: the world model's point estimate of the official
  test accuracy of the checkpoint this candidate's recipe produces.
- `rank`: candidates ordered by that estimate (1 = best).
- `nearest_prior_recipes`: the most similar recipes among the prior runs, each
  with its official accuracy and a one-line description. These are the concrete
  evidence behind the estimate; open `prior_runs/<run>/cards/<card>.json` for details.

## How the model was built and how reliable it is

Model: `{variant}` — {family}. It was fitted on the {n_train} labeled
experiments of the {n_runs} prior runs in `prior_runs/` (recipes only: data,
method, hyperparameters, launch command, ancestors, scripts; never any score),
predicting each experiment's official test accuracy.

Reliability, measured by holding out whole prior runs inside the training set
(so these numbers are for recipes from scientists the model never saw):

- mean absolute error of the accuracy estimate: **{mae:.3f}** (accuracies here range {lo:.2f}-{hi:.2f}, sd {sd:.3f})
- rank correlation with the true accuracy: **{spearman:.2f}**
- when two sibling experiments from the same parent differ by >=1 point, the model
  ranks them correctly **{pair:.0%}** of the time; picking the best of a sibling
  set: **{top1:.0%}** (random {random_top1:.0%}), leaving {regret:.3f} accuracy on the table on average (random {random_regret:.3f}).

## How to use it well

- Use the ranking, not just the point estimates. The absolute error ({mae:.3f}) is
  large compared with typical sibling gaps, yet the model still orders siblings
  correctly {pair:.0%} of the time, because siblings share most of their recipe and
  the model's errors cancel. Small predicted gaps are still informative.
- Treat the model's top-ranked candidate as the default choice. Move away from it
  only with concrete evidence you can cite; never pick a candidate the model ranks
  in the bottom half without a specific defect found in the higher-ranked ones.
- The model is especially good at spotting recipes that resemble prior failures
  (collapsed runs, over-long training, bad data); trust low predictions.
- The model sees the recipe as registered, not what the code actually does. It
  cannot detect bugs, prompt/answer-format mismatches, evaluation-protocol
  errors, or truncation problems. Check the candidate scripts for those and
  override the model when you find a concrete defect.
- The model has no information about this scientist's run; neither do you.
- Prior-run evidence the model has not seen well (recipes unlike anything in
  `nearest_prior_recipes`, low similarity) deserves more of your own judgment.
"""

FAMILY_DESC = {
    "hgb_structured": "gradient-boosted trees over ~130 structured recipe features (method family, learning rate, epochs, batch, sequence length, data sources and counts, decoding settings, ancestor chain statistics, code markers)",
    "rf_structured": "random forest over ~130 structured recipe features",
    "et_structured": "extra-trees over ~130 structured recipe features",
    "et_structured_text": "extra-trees over structured recipe features plus a TF-IDF embedding of the recipe text and scripts",
    "hgb_structured_text": "gradient-boosted trees over structured recipe features plus a TF-IDF embedding of the recipe text and scripts",
    "ridge_tfidf": "ridge regression over TF-IDF of the recipe text and scripts",
    "ridge_tfidf_structured": "ridge regression over TF-IDF plus structured features",
    "knn5_tfidf": "5-nearest-neighbour regression over TF-IDF of the recipe text",
    "ens_et_hgb_ridge": "average of extra-trees, gradient boosting and ridge models over structured and text features",
    "pair_hgb_sibling": "gradient-boosted pairwise ranker trained on sibling comparisons, on top of a pointwise gradient-boosting model",
    "pair_et_sibling": "extra-trees pairwise ranker trained on sibling comparisons, on top of a pointwise gradient-boosting model",
    "pair_hgb_withinrun": "gradient-boosted pairwise ranker trained on all within-run comparisons, on top of a pointwise model",
    "pair_hgb_sibling_text": "gradient-boosted pairwise ranker over structured and text features",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    ap.add_argument("--sets", type=Path, default=Path("data/analysis/wm_study/rpm"))
    ap.add_argument("--cv", type=Path, default=Path("data/analysis/wm_study/wm_cv2"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/rpm/wm_tool"))
    ap.add_argument("--readme-version", type=int, default=2)
    ap.add_argument("--variant", default="nested", help="variant name, or 'nested' to use the per-fold inner-CV choice from --cv")
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k", "aime2025"])
    ap.add_argument("--neighbors", type=int, default=5)
    args = ap.parse_args()
    registry = [json.loads(l) for l in (args.sets / "registry.jsonl").read_text().splitlines() if l.strip()]
    variants = {v.name: v for v in make_variants()}
    for benchmark in args.benchmarks:
        rows = [r for r in load_cards(args.cards, benchmark) if r["lineage_complete"]]
        by_id = {r["example_id"]: r for r in rows}
        fold_of = make_folds(rows, 8)
        nested = json.loads((args.cv / benchmark / "metrics.json").read_text())["nested_choice"] if args.variant == "nested" else {}
        sets_by_fold = defaultdict(list)
        for e in registry:
            if e["benchmark"] == benchmark:
                sets_by_fold[e["fold"]].append(e)
        for fold in sorted(sets_by_fold):
            vname = nested.get(str(fold), nested.get(fold)) if args.variant == "nested" else args.variant
            v = variants[vname]
            train = [r for r in rows if fold_of[r["cell_id"]] != fold]
            ytr = np.array([r["official_accuracy"] for r in train])
            # reliability by inner CV on the training runs
            inner = make_folds(train, k=4, seed=fold + 100)
            ip = {}
            for ifold in sorted(set(inner.values())):
                itr = [r for r in train if inner[r["cell_id"]] != ifold]
                ite = [r for r in train if inner[r["cell_id"]] == ifold]
                v.fit(itr, np.array([r["official_accuracy"] for r in itr]))
                for r, p in zip(ite, v.predict(ite)):
                    ip[r["example_id"]] = float(p)
            pm = point_metrics([by_id[k]["official_accuracy"] for k in ip], [ip[k] for k in ip])
            sm = selection_metrics(train, ip)
            rnd = selection_metrics(train, {k: 0.0 for k in ip})  # ties -> half credit; use random_acc/top1 from set stats
            v.fit(train, ytr)
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b[\w][\w.\-=]*\b")
            T = vec.fit_transform([recipe_text(r) for r in train])
            for e in sets_by_fold[fold]:
                set_json = json.loads((args.sets / "sets" / benchmark / e["set_id"] / "set.json").read_text())
                cands = [(L, by_id[ex]) for L, ex in set_json["letter_to_example"].items()]
                preds = v.predict([c for _, c in cands])
                order = sorted(range(len(cands)), key=lambda i: -preds[i])
                rank = {cands[i][0]: k + 1 for k, i in enumerate(order)}
                Q = vec.transform([recipe_text(c) for _, c in cands])
                sims = (T @ Q.T).toarray()
                out = {"model": vname, "candidates": {}}
                for i, (L, c) in enumerate(cands):
                    top = np.argsort(-sims[:, i])[: args.neighbors]
                    out["candidates"][L] = {
                        "predicted_official_accuracy": round(float(preds[i]), 4),
                        "rank": rank[L],
                        "nearest_prior_recipes": [
                            {"run": train[j]["cell_id"], "card": train[j]["card_id"], "official_accuracy": round(train[j]["official_accuracy"], 4), "similarity": round(float(sims[j, i]), 3), "recipe": short_recipe(train[j])}
                            for j in top
                        ],
                    }
                d = args.out / benchmark / e["set_id"]
                d.mkdir(parents=True, exist_ok=True)
                (d / "predictions.json").write_text(json.dumps(out, indent=1))
                ys = ytr
                (d / "README.md").write_text(README.format(
                    variant=vname, family=FAMILY_DESC.get(vname, vname), n_train=len(train), n_runs=len({r["cell_id"] for r in train}),
                    mae=pm["mae"], lo=float(ys.min()), hi=float(ys.max()), sd=float(ys.std()), spearman=pm["spearman"],
                    pair=sm["pairwise_acc"], top1=sm["top1_acc"], random_top1=1.0 / np.mean([len(s["cards"]) for s in decision_sets(train)]),
                    regret=sm["regret"], random_regret=sm["oracle_acc"] - sm["random_acc"], mae_half=pm["mae"] / 2))
            print(f"{benchmark} fold {fold}: {vname} inner MAE {pm['mae']:.3f} pair {sm['pairwise_acc']:.3f}; wrote {len(sets_by_fold[fold])} sets")


if __name__ == "__main__":
    main()
