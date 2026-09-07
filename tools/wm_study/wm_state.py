"""One-step look-ahead world model: history as a recursively updated state.

predict(child) = f(parent accuracy, state(parent), target step, interactions)
state(child)   = update(state(parent), step, parent accuracy)

Compared on the same run-held-out folds against the flat recipe model.
Usage: python -m tools.wm_study.wm_state --benchmarks gsm8k aime2025
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.wm_study.wm import _num, load_cards, make_folds, point_metrics, run_weights, selection_metrics, step_features, TreeStructured

BASE_ACC = {"gsm8k": 0.045, "aime2025": 0.05}
FAMS = ("sft", "rft", "merge", "decoding", "evaluation", "grpo", "rl", "distill", "other")
GOLD = re.compile(r"(?i)openai/gsm8k|gsm8k \(|metamath|openmath|openr1|numina|aime|math-ai|hendrycks")
SELF = re.compile(r"(?i)derived|synthetic|self")


def _sources(step):
    return [str(d.get("source") or "") + "|" + str(d.get("path") or "") for d in (step["setup"].get("data") or [])]


def _fmt(step):
    m = step["setup"].get("method") or {}
    t = str(m.get("target_format") or "").lower()
    return tuple(tok for tok in ("answer:", "####", "boxed", "<end_of_turn>", "<|im_end|>", "think") if tok in t)


def empty_state(benchmark):
    return {"acc": BASE_ACC[benchmark], "acc_prev": np.nan, "n_steps": 0, "ex_total": 0.0, "ex_gold": 0.0, "ex_self": 0.0, "epochs_total": 0.0, "steps_total": 0.0, "tokens_total": 0.0,
            **{f"n_{f}": 0.0 for f in FAMS}, "last_lr": np.nan, "last_epochs": np.nan, "last_msl": np.nan, "last_family": "base", "n_format_changes": 0.0, "fmt": (), "seen_sources": set(),
            "sampler_acc": np.nan, "n_merges": 0.0, "last_gain": np.nan, "max_acc": BASE_ACC[benchmark]}


def update(state, step, child_acc, sampler_acc=np.nan):
    s = dict(state)
    s["seen_sources"] = set(state["seen_sources"])
    h = (step["setup"].get("method") or {}).get("hyperparams") or {}
    data = step["setup"].get("data") or []
    n = sum((_num(d.get("n_examples")) or 0) for d in data)
    ep = _num(h.get("epochs")) or 0.0
    bs, ga = _num(h.get("batch_size")) or 0, _num(h.get("grad_accum")) or 1
    msl = _num(h.get("max_seq_len")) or 0
    s["n_steps"] += 1
    s["ex_total"] += n * max(ep, 1e-9) if ep else n
    for d in data:
        src = str(d.get("source") or "")
        k = (_num(d.get("n_examples")) or 0) * (ep or 1)
        if SELF.search(src):
            s["ex_self"] += k
        elif GOLD.search(src):
            s["ex_gold"] += k
    s["epochs_total"] += ep
    s["steps_total"] += (n * ep / (bs * ga)) if bs and ep else 0.0
    s["tokens_total"] += n * ep * msl
    fam = step["family"] if step["family"] in FAMS else "other"
    s[f"n_{fam}"] += 1
    if fam == "merge":
        s["n_merges"] += 1
    lr = _num(h.get("lr"))
    if lr:
        s["last_lr"] = math.log10(lr)
    if ep:
        s["last_epochs"] = ep
    if msl:
        s["last_msl"] = msl
    s["last_family"] = fam
    f = _fmt(step)
    if f and f != state["fmt"]:
        s["n_format_changes"] += 1
        s["fmt"] = f
    s["seen_sources"].update(_sources(step))
    if not math.isnan(sampler_acc):
        s["sampler_acc"] = sampler_acc
    s["acc_prev"] = state["acc"]
    s["acc"] = child_acc
    s["last_gain"] = child_acc - state["acc"] if not math.isnan(child_acc) else np.nan
    s["max_acc"] = max(state["max_acc"], child_acc) if not math.isnan(child_acc) else state["max_acc"]
    return s


def merge_states(states):
    s = dict(states[0])
    keys = [k for k, v in s.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for k in keys:
        vals = [st[k] for st in states if not (isinstance(st[k], float) and math.isnan(st[k]))]
        s[k] = float(np.mean(vals)) if vals else np.nan
    s["acc"] = max(st["acc"] for st in states)
    s["seen_sources"] = set().union(*(st["seen_sources"] for st in states))
    return s


def build_states(cards_by_id, benchmark):
    """Compute state(card) = state after applying the card's step, for every card in a run."""
    memo = {}

    def state_of(ex_id, stack=()):
        if ex_id in memo:
            return memo[ex_id]
        c = cards_by_id[ex_id]
        cell = c["cell_id"]
        if ex_id in stack:
            return {"parent_state": empty_state(benchmark), "state": empty_state(benchmark), "sampler_acc": np.nan}
        parents = [f"{cell}/{p}" for p in c["parents"] if f"{cell}/{p}" in cards_by_id]
        if not parents:
            pstate = empty_state(benchmark)
        elif len(parents) == 1:
            pstate = state_of(parents[0], stack + (ex_id,))["state"]
        else:
            pstate = merge_states([state_of(p, stack + (ex_id,))["state"] for p in parents])
        deps = [f"{cell}/{d}" for d in c["data_deps"] if f"{cell}/{d}" in cards_by_id]
        sampler = max([cards_by_id[d]["official_accuracy"] for d in deps if cards_by_id[d]["official_accuracy"] is not None], default=np.nan)
        step = c["recipe"][-1]
        acc = c["official_accuracy"] if c["official_accuracy"] is not None else np.nan
        memo[ex_id] = {"parent_state": pstate, "state": update(pstate, step, acc, sampler), "sampler_acc": sampler}
        return memo[ex_id]

    for ex in cards_by_id:
        state_of(ex)
    return memo


def features(card, pstate, sampler_acc, use_parent_acc=True):
    step = card["recipe"][-1]
    f = {("t_" + k[2:]): v for k, v in step_features(step, "t_").items()}
    if use_parent_acc:
        f["parent_acc"] = pstate["acc"]
        f["grandparent_acc"] = pstate["acc_prev"]
        f["last_gain"] = pstate["last_gain"]
        f["max_acc_so_far"] = pstate["max_acc"]
        f["sampler_acc"] = sampler_acc
        f["sampler_minus_parent"] = sampler_acc - pstate["acc"] if not math.isnan(sampler_acc) else np.nan
    for k in ("n_steps", "ex_total", "ex_gold", "ex_self", "epochs_total", "steps_total", "tokens_total", "last_lr", "last_epochs", "last_msl", "n_format_changes", "n_merges") + tuple(f"n_{x}" for x in FAMS):
        v = pstate[k]
        f["s_" + k] = math.log1p(v) if k in ("ex_total", "ex_gold", "ex_self", "steps_total", "tokens_total") and v == v else v
    for fam in FAMS + ("base",):
        f[f"s_last_{fam}"] = float(pstate["last_family"] == fam)
    f["s_parent_is_base"] = float(pstate["n_steps"] == 0)
    # interactions between history and the current step
    srcs = _sources(step)
    f["i_novelty"] = (sum(s not in pstate["seen_sources"] for s in srcs) / len(srcs)) if srcs else np.nan
    h = (step["setup"].get("method") or {}).get("hyperparams") or {}
    n = sum((_num(d.get("n_examples")) or 0) for d in (step["setup"].get("data") or []))
    ep = _num(h.get("epochs")) or 0.0
    f["i_dose_ratio"] = math.log1p(n * ep) - math.log1p(pstate["ex_total"]) if pstate["ex_total"] else math.log1p(n * ep)
    f["i_format_change"] = float(bool(_fmt(step)) and _fmt(step) != pstate["fmt"] and bool(pstate["fmt"]))
    f["i_same_family"] = float(step["family"] == pstate["last_family"])
    f["i_lr_vs_last"] = (math.log10(_num(h.get("lr"))) - pstate["last_lr"]) if _num(h.get("lr")) and pstate["last_lr"] == pstate["last_lr"] else np.nan
    return f


class StateModel:
    def __init__(self, use_parent_acc=True, target="abs", with_flat=False):
        self.use_parent_acc, self.target, self.with_flat = use_parent_acc, target, with_flat

    def _X(self, rows, memo):
        from tools.wm_study.wm import structured_features
        feats = [features(r, memo[r["example_id"]]["parent_state"], memo[r["example_id"]]["sampler_acc"], self.use_parent_acc) for r in rows]
        if self.with_flat:
            feats = [{**f, **{"f_" + k: v for k, v in structured_features(r).items()}} for f, r in zip(feats, rows)]
        if not hasattr(self, "keys"):
            self.keys = sorted(feats[0])
        return np.array([[float(f.get(k, np.nan)) if f.get(k) is not None else np.nan for k in self.keys] for f in feats])

    def fit(self, rows, memo):
        X = self._X(rows, memo)
        self.all_nan = np.isnan(X).all(axis=0)
        X[:, self.all_nan] = 0.0
        y = np.array([r["official_accuracy"] for r in rows])
        w = run_weights(rows)
        pacc = np.array([memo[r["example_id"]]["parent_state"]["acc"] for r in rows])
        self.pacc_fill = float(np.nanmedian(pacc))
        if self.target == "delta":
            keep = ~np.isnan(pacc)
            X, y, w = X[keep], y[keep] - pacc[keep], w[keep]
        self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=0)
        self.m.fit(X, y, sample_weight=w)
        return self

    def predict(self, rows, memo):
        X = self._X(rows, memo)
        X[:, self.all_nan] = 0.0
        p = self.m.predict(X)
        if self.target == "delta":
            pacc = np.array([memo[r["example_id"]]["parent_state"]["acc"] for r in rows])
            p = p + np.where(np.isnan(pacc), self.pacc_fill, pacc)
        return np.clip(p, 0, 1)


def within_run_pairs(rows, pred):
    """Order any two labeled checkpoints of the same held-out run (different parents allowed)."""
    by = defaultdict(list)
    for r in rows:
        by[r["cell_id"]].append(r)
    hit = tot = 0.0
    for g in by.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                if abs(a["official_accuracy"] - b["official_accuracy"]) < 0.01:
                    continue
                tot += 1
                pa, pb = pred[a["example_id"]], pred[b["example_id"]]
                hit += 0.5 if pa == pb else float((pa > pb) == (a["official_accuracy"] > b["official_accuracy"]))
    return hit / max(tot, 1), int(tot)


def gain_sign_acc(rows, pred, memo):
    hit = tot = 0
    for r in rows:
        pa = memo[r["example_id"]]["parent_state"]["acc"]
        d = r["official_accuracy"] - pa
        if abs(d) < 0.01:
            continue
        tot += 1
        hit += (pred[r["example_id"]] - pa > 0) == (d > 0)
    return hit / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k", "aime2025"])
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/wm_state"))
    args = ap.parse_args()
    for benchmark in args.benchmarks:
        all_rows = load_cards(args.cards, benchmark, labeled_only=False)
        by_id = {r["example_id"]: r for r in all_rows}
        memo = build_states(by_id, benchmark)
        rows = [r for r in all_rows if r["official_accuracy"] is not None and r["lineage_complete"]]
        fold_of = make_folds(rows, 8)
        y = {r["example_id"]: r["official_accuracy"] for r in rows}
        variants = {
            "flat recipe (current)": ("flat", None),
            "flat recipe + parent acc": ("flat+pacc", None),
            "state, no parent acc": ("state", StateModel(use_parent_acc=False)),
            "state + parent acc": ("state", StateModel(use_parent_acc=True)),
            "state + parent acc, predict delta": ("state", StateModel(use_parent_acc=True, target="delta")),
            "flat + state + parent acc": ("state", StateModel(use_parent_acc=True, with_flat=True)),
            "flat + state, no parent acc": ("state", StateModel(use_parent_acc=False, with_flat=True)),
            "parent acc as the guess": ("guess", None),
        }
        print(f"\n== {benchmark}: {len(rows)} labeled checkpoints, {len(fold_of)} runs")
        print("| model | MAE | Spearman | sibling pairwise | sibling regret | chosen acc | within-run pairwise | gain-sign acc |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        results = {}
        for name, (kind, model) in variants.items():
            pred = {}
            for k in range(8):
                tr = [r for r in rows if fold_of[r["cell_id"]] != k]
                te = [r for r in rows if fold_of[r["cell_id"]] == k]
                if kind == "guess":
                    for r in te:
                        pred[r["example_id"]] = memo[r["example_id"]]["parent_state"]["acc"]
                    continue
                if kind.startswith("flat"):
                    import tools.wm_study.wm as W
                    orig = W.structured_features
                    if kind == "flat+pacc":
                        W.structured_features = lambda c, o=orig: {**o(c), "parent_acc": memo[c["example_id"]]["parent_state"]["acc"]}
                    v = TreeStructured("h", "hgb", False).fit(tr, np.array([r["official_accuracy"] for r in tr]))
                    p = v.predict(te)
                    W.structured_features = orig
                else:
                    v = StateModel(model.use_parent_acc, model.target, model.with_flat).fit(tr, memo)
                    p = v.predict(te, memo)
                for r, pp in zip(te, p):
                    pred[r["example_id"]] = float(pp)
            pm = point_metrics([y[i] for i in pred], [pred[i] for i in pred])
            sm = selection_metrics(rows, pred)
            wr, nwr = within_run_pairs(rows, pred)
            gs, ngs = gain_sign_acc(rows, pred, memo)
            results[name] = {"point": pm, "selection": sm, "within_run_pairwise": wr, "gain_sign": gs}
            print(f"| {name} | {pm['mae']:.3f} | {pm['spearman']:.2f} | {sm['pairwise_acc']:.3f} | {sm['regret']:.3f} | {sm['chosen_acc']:.3f} | {wr:.3f} (n={nwr}) | {gs:.3f} (n={ngs}) |")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{benchmark}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
