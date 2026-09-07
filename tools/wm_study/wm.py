"""World-model variants: recipe chain -> official accuracy, evaluated run-held-out.

Usage: python -m tools.wm_study.wm --cards data/analysis/wm_study/cards.jsonl \
           --out data/analysis/wm_study/wm_cv
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

import os

FEATURE_VERSION = int(os.environ.get("WM_FEATURE_VERSION", "1"))
FAMILIES = ("sft", "rft", "merge", "decoding", "evaluation", "grpo", "rl", "distill", "other", "unknown")
SOURCE_TOKENS = ("gsm8k", "openmathinstruct", "openmathreasoning", "openr1", "metamath", "numina", "math", "aime", "derived", "synthetic", "self", "deepseek", "distill", "orca", "hendrycks")
SCRIPT_TOKENS = ("completion_only", "assistant_only", "lora", "liger", "vllm", "packing", "gradient_checkpointing", "flash_attention", "chat_template", "generation_config", "do_sample", "temperature", "max_new_tokens", "repetition_penalty", "<end_of_turn>", "boxed", "ANSWER:", "####", "think")

# ----------------------------------------------------------------------------- data


def load_cards(path, benchmark=None, labeled_only=True):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if benchmark:
        rows = [r for r in rows if r["benchmark"] == benchmark]
    if labeled_only:
        rows = [r for r in rows if r["official_accuracy"] is not None]
    return rows


def make_folds(rows, k=8, seed=20260905):
    """Assign every run (cell) to one of k folds, stratified by scientist model."""
    cells = sorted({(r["cell_id"], r["scientist_model"]) for r in rows})
    rng = random.Random(seed)
    fold_of = {}
    by_model = defaultdict(list)
    for cell, model in cells:
        by_model[model].append(cell)
    counter = 0
    for model in sorted(by_model):
        group = by_model[model]
        rng.shuffle(group)
        for cell in group:
            fold_of[cell] = counter % k
            counter += 1
    return fold_of


# ----------------------------------------------------------------------------- features


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", str(v))
        if not m:
            return None
        x = float(m.group())
    return x if math.isfinite(x) else None


def step_features(step, prefix):
    s = step["setup"] or {}
    m = s.get("method") or {}
    h = m.get("hyperparams") or {}
    f = {}
    for fam in FAMILIES:
        f[f"{prefix}fam_{fam}"] = float(step["family"] == fam)
    lr = _num(h.get("lr"))
    f[f"{prefix}log_lr"] = math.log10(lr) if lr and lr > 0 else np.nan
    f[f"{prefix}epochs"] = _num(h.get("epochs"))
    bs, ga = _num(h.get("batch_size")), _num(h.get("grad_accum"))
    f[f"{prefix}eff_batch"] = (bs or 0) * (ga or 1) if bs else np.nan
    f[f"{prefix}max_seq_len"] = _num(h.get("max_seq_len"))
    f[f"{prefix}warmup"] = _num(h.get("warmup"))
    f[f"{prefix}weight_decay"] = _num(h.get("weight_decay"))
    f[f"{prefix}temperature"] = _num(h.get("temperature"))
    f[f"{prefix}top_p"] = _num(h.get("top_p"))
    f[f"{prefix}max_tokens"] = _num(h.get("max_tokens"))
    f[f"{prefix}rep_penalty"] = _num(h.get("repetition_penalty"))
    peft = str(m.get("peft") or "none").lower()
    f[f"{prefix}peft_lora"] = float("lora" in peft)
    prec = str(h.get("precision") or "").lower()
    f[f"{prefix}bf16"] = float("bf16" in prec)
    data = s.get("data") or []
    n_ex = [(_num(d.get("n_examples")) or 0) for d in data]
    f[f"{prefix}log_n_examples"] = math.log1p(sum(n_ex))
    f[f"{prefix}n_sources"] = float(len(data))
    f[f"{prefix}max_mix"] = max([(_num(d.get("mixture_weight")) or 0) for d in data], default=0.0)
    src_text = " ".join(str(d.get("source") or "") + " " + str(d.get("selection") or "") + " " + str(d.get("path") or "") for d in data).lower()
    for tok in SOURCE_TOKENS:
        f[f"{prefix}src_{tok}"] = float(tok in src_text)
    f[f"{prefix}contam_passed"] = float(any(str(d.get("contamination_check") or "").lower().startswith("pass") for d in data))
    f[f"{prefix}planned_h"] = _num((s.get("budget") or {}).get("planned_h"))
    prog = s.get("progress") or {}
    tot = _num(prog.get("total"))
    f[f"{prefix}log_steps"] = math.log1p(tot) if tot else np.nan
    argv = " ".join(map(str, (s.get("command") or {}).get("argv") or [])).lower()
    scripts = " ".join(sc["content"] for sc in step.get("scripts") or []).lower()
    for tok in SCRIPT_TOKENS:
        f[f"{prefix}code_{tok}"] = float(tok.lower() in scripts or tok.lower() in argv)
    f[f"{prefix}script_chars"] = math.log1p(len(scripts))
    other = str(h.get("other") or "").lower()
    f[f"{prefix}other_greedy"] = float("greedy" in other or "do_sample=false" in other)
    if FEATURE_VERSION >= 2:
        # training dose: the mechanism behind most collapses in the prior-run cards
        n_total = sum(n_ex)
        ep = f[f"{prefix}epochs"] or 0.0
        eb = f[f"{prefix}eff_batch"] if f[f"{prefix}eff_batch"] == f[f"{prefix}eff_batch"] else 0.0
        steps = n_total * ep / eb if eb else 0.0
        f[f"{prefix}log_steps_est"] = math.log1p(steps)
        f[f"{prefix}log_examples_seen"] = math.log1p(n_total * ep)
        f[f"{prefix}log_lr_x_steps"] = math.log1p((lr or 0.0) * steps * 1e4)
        msl = f[f"{prefix}max_seq_len"] or 0.0
        f[f"{prefix}log_tokens_est"] = math.log1p(n_total * ep * msl)
        f[f"{prefix}code_eos"] = float("eos" in scripts)
        f[f"{prefix}code_pad_token"] = float("pad_token" in scripts)
        f[f"{prefix}code_stop"] = float("stop" in scripts or "stop_strings" in scripts)
        f[f"{prefix}code_dedup"] = float("dedup" in scripts or "set(" in scripts)
        f[f"{prefix}code_filter_correct"] = float("correct" in scripts and ("filter" in scripts or "keep" in scripts))
        f[f"{prefix}code_max_new_tokens_val"] = _num(re.search(r"max_new_tokens\D{0,5}(\d+)", scripts).group(1)) if re.search(r"max_new_tokens\D{0,5}(\d+)", scripts) else np.nan
        f[f"{prefix}n_mix_derived"] = float(sum("derived" in str(d.get("source") or "").lower() or "synthetic" in str(d.get("source") or "").lower() for d in data))
        f[f"{prefix}argv_len"] = float(len((s.get("command") or {}).get("argv") or []))
    return f


def structured_features(card):
    chain = card["recipe"]
    target = chain[-1]
    f = step_features(target, "t_")
    ancestors = [s for s in chain[:-1] if s["relation"] == "weights"]
    data_steps = [s for s in chain[:-1] if s["relation"] == "data"]
    f["depth"] = float(card["depth"])
    f["n_weight_ancestors"] = float(len(ancestors))
    f["n_data_deps"] = float(len(data_steps))
    for fam in FAMILIES:
        f[f"anc_count_{fam}"] = float(sum(s["family"] == fam for s in ancestors))
    if ancestors:
        parent = ancestors[-1]
        f.update(step_features(parent, "p_"))
        anc_n = sum(sum((_num(d.get("n_examples")) or 0) for d in (s["setup"].get("data") or [])) for s in ancestors)
        f["anc_log_n_examples"] = math.log1p(anc_n)
        f["anc_epochs_sum"] = sum((_num(((s["setup"].get("method") or {}).get("hyperparams") or {}).get("epochs")) or 0) for s in ancestors)
    else:
        for k, v in step_features(target, "p_").items():
            f[k] = np.nan
        f["anc_log_n_examples"] = 0.0
        f["anc_epochs_sum"] = 0.0
    f["n_parents"] = float(len(card["parents"]))
    base = str((chain[0]["setup"] or {}).get("base_model") or card.get("base_model") or "").lower()
    for tok in ("gemma", "qwen3-4b", "qwen3-1.7b", "smollm"):
        f[f"base_{tok}"] = float(tok in base)
    if FEATURE_VERSION >= 2:
        chain_steps = 0.0
        chain_examples = 0.0
        for st in ancestors + [target]:
            sf = step_features(st, "x_")
            chain_steps += math.expm1(sf["x_log_steps_est"])
            chain_examples += math.expm1(sf["x_log_examples_seen"])
        f["chain_log_steps"] = math.log1p(chain_steps)
        f["chain_log_examples_seen"] = math.log1p(chain_examples)
        f["target_share_of_chain_steps"] = math.expm1(f["t_log_steps_est"]) / chain_steps if chain_steps else 0.0
        f["family_same_as_parent"] = float(bool(ancestors) and ancestors[-1]["family"] == target["family"])
    return f


_PATH = re.compile(r"/[\w./\-]+")
_HEX = re.compile(r"\b[0-9a-f]{12,}\b")


def recipe_text(card, include_scripts=True):
    parts = []
    for step in card["recipe"]:
        s = json.dumps(step["setup"], ensure_ascii=False, sort_keys=True)
        s = _PATH.sub(" path ", s)
        s = _HEX.sub(" hash ", s)
        parts.append(f"STEP {step['relation']} family={step['family']} {s}")
        if include_scripts:
            for sc in step.get("scripts") or []:
                parts.append(f"SCRIPT {sc['name']}\n{sc['content'][:20000]}")
    return "\n".join(parts)


class FeatureSpace:
    """Fit feature transforms on training rows only."""

    def __init__(self, text=True, svd_dim=64):
        self.text = text
        self.svd_dim = svd_dim

    def fit(self, rows):
        feats = [structured_features(r) for r in rows]
        self.keys = sorted(feats[0])
        X = np.array([[f.get(k, np.nan) for k in self.keys] for f in feats], dtype=float)
        self.medians = np.nanmedian(X, axis=0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        if self.text:
            self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b[\w][\w.\-=]*\b")
            T = self.vec.fit_transform([recipe_text(r) for r in rows])
            self.svd = TruncatedSVD(n_components=min(self.svd_dim, T.shape[0] - 1, T.shape[1] - 1), random_state=0)
            self.svd.fit(T)
        return self

    def structured(self, rows):
        feats = [structured_features(r) for r in rows]
        X = np.array([[f.get(k, np.nan) for k in self.keys] for f in feats], dtype=float)
        return np.where(np.isnan(X), self.medians, X)

    def tfidf(self, rows):
        return self.vec.transform([recipe_text(r) for r in rows])

    def dense_text(self, rows):
        return self.svd.transform(self.tfidf(rows))


# ----------------------------------------------------------------------------- variants


def run_weights(rows):
    counts = Counter(r["cell_id"] for r in rows)
    return np.array([r.get("weight_scale", 1.0) / counts[r["cell_id"]] for r in rows])


class Variant:
    def __init__(self, name):
        self.name = name

    def fit(self, rows, y):
        raise NotImplementedError

    def predict(self, rows):
        raise NotImplementedError


class TreeStructured(Variant):
    def __init__(self, name="et_structured", kind="et", with_text=False):
        super().__init__(name)
        self.kind, self.with_text = kind, with_text

    def _model(self):
        if self.kind == "et":
            return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=4)
        if self.kind == "rf":
            return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=4)
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=0)

    def _X(self, rows):
        X = self.fs.structured(rows)
        if self.with_text:
            X = np.hstack([X, self.fs.dense_text(rows)])
        return X

    def fit(self, rows, y):
        self.fs = FeatureSpace(text=self.with_text).fit(rows)
        self.m = self._model()
        self.m.fit(self._X(rows), y, sample_weight=run_weights(rows))
        return self

    def predict(self, rows):
        return np.clip(self.m.predict(self._X(rows)), 0, 1)


class RidgeText(Variant):
    def __init__(self, name="ridge_tfidf", alpha=1.0, with_structured=True):
        super().__init__(name)
        self.alpha, self.with_structured = alpha, with_structured

    def _X(self, rows):
        T = self.fs.tfidf(rows)
        if self.with_structured:
            S = self.fs.structured(rows)
            S = (S - self.mu) / self.sd
            return hstack([T, csr_matrix(S)]).tocsr()
        return T

    def fit(self, rows, y):
        self.fs = FeatureSpace(text=True).fit(rows)
        S = self.fs.structured(rows)
        self.mu, self.sd = S.mean(0), S.std(0) + 1e-6
        self.m = Ridge(alpha=self.alpha)
        self.m.fit(self._X(rows), y, sample_weight=run_weights(rows))
        return self

    def predict(self, rows):
        return np.clip(self.m.predict(self._X(rows)), 0, 1)


class KNNText(Variant):
    def __init__(self, name="knn_tfidf", k=5):
        super().__init__(name)
        self.k = k

    def fit(self, rows, y):
        self.fs = FeatureSpace(text=True).fit(rows)
        self.m = KNeighborsRegressor(n_neighbors=min(self.k, len(rows)), metric="cosine", weights="distance")
        self.m.fit(self.fs.tfidf(rows), y)
        return self

    def predict(self, rows):
        return np.clip(self.m.predict(self.fs.tfidf(rows)), 0, 1)


class Ensemble(Variant):
    def __init__(self, name, members):
        super().__init__(name)
        self.members = members

    def fit(self, rows, y):
        for m in self.members:
            m.fit(rows, y)
        return self

    def predict(self, rows):
        return np.mean([m.predict(rows) for m in self.members], axis=0)


class PairwiseRanker(Variant):
    """Sibling-trained pairwise preference model: P(a beats b) from [a-b, (a+b)/2].

    Scores a candidate by its mean predicted win probability against all siblings
    in its decision set; outside a set (single card) it falls back to a pointwise
    model so the score is still on the accuracy scale.
    """

    def __init__(self, name="pair_hgb", kind="hgb", with_text=False, all_within_run=False):
        super().__init__(name)
        self.kind, self.with_text, self.all_within_run = kind, with_text, all_within_run

    def _clf(self):
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
        if self.kind == "et":
            return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=4)
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=0)

    def _X(self, rows):
        X = self.fs.structured(rows)
        if self.with_text:
            X = np.hstack([X, self.fs.dense_text(rows)])
        return X

    def fit(self, rows, y):
        self.fs = FeatureSpace(text=self.with_text).fit(rows)
        X = self._X(rows)
        idx = {r["example_id"]: i for i, r in enumerate(rows)}
        groups = defaultdict(list)
        for r in rows:
            key = r["cell_id"] if self.all_within_run else (r["cell_id"], tuple(r["parents"]))
            groups[key].append(r)
        A, B, lab, w = [], [], [], []
        for g in groups.values():
            for i in range(len(g)):
                for j in range(len(g)):
                    if i == j or abs(g[i]["official_accuracy"] - g[j]["official_accuracy"]) < 0.01:
                        continue
                    A.append(idx[g[i]["example_id"]]); B.append(idx[g[j]["example_id"]])
                    lab.append(int(g[i]["official_accuracy"] > g[j]["official_accuracy"]))
                    w.append(1.0 / (len(g) * (len(g) - 1)))
        self.point = TreeStructured("pt", "hgb", False).fit(rows, y)
        if len(set(lab)) < 2:
            self.clf = None
            return self
        Xa, Xb = X[A], X[B]
        F = np.hstack([Xa - Xb, (Xa + Xb) / 2])
        self.clf = self._clf().fit(F, np.array(lab), sample_weight=np.array(w))
        return self

    def _pref(self, Xa, Xb):
        F = np.hstack([Xa - Xb, (Xa + Xb) / 2])
        G = np.hstack([Xb - Xa, (Xa + Xb) / 2])
        return (self.clf.predict_proba(F)[:, 1] + 1 - self.clf.predict_proba(G)[:, 1]) / 2

    def predict(self, rows):
        base = self.point.predict(rows)
        if self.clf is None:
            return base
        X = self._X(rows)
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[(r["cell_id"], tuple(r["parents"]))].append(i)
        out = base.copy()
        for members in groups.values():
            if len(members) < 2:
                continue
            for i in members:
                others = [j for j in members if j != i]
                p = self._pref(np.repeat(X[i:i + 1], len(others), 0), X[others]).mean()
                # keep the accuracy scale: shift the set's pointwise mean by the preference
                out[i] = float(np.clip(base[members].mean() + (p - 0.5) * 0.3, 0, 1))
        return out


class SeedEnsemble(Variant):
    """Average of several structured tree models with different seeds/families."""

    def __init__(self, name="ens_struct_seeds", with_text=False):
        super().__init__(name)
        self.with_text = with_text

    def fit(self, rows, y):
        self.fs = FeatureSpace(text=self.with_text).fit(rows)
        X = self.fs.structured(rows)
        if self.with_text:
            X = np.hstack([X, self.fs.dense_text(rows)])
        w = run_weights(rows)
        self.models = []
        for seed in range(5):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=seed)
            self.models.append(m.fit(X, y, sample_weight=w))
        self.models.append(ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=4).fit(X, y, sample_weight=w))
        self.models.append(RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=4).fit(X, y, sample_weight=w))
        return self

    def predict(self, rows):
        X = self.fs.structured(rows)
        if self.with_text:
            X = np.hstack([X, self.fs.dense_text(rows)])
        return np.clip(np.mean([m.predict(X) for m in self.models], axis=0), 0, 1)


class TrainMean(Variant):
    def fit(self, rows, y):
        self.mu = float(np.average(y, weights=run_weights(rows)))
        return self

    def predict(self, rows):
        return np.full(len(rows), self.mu)


def make_variants():
    if os.environ.get("WM_VARIANTS") == "compact":
        return [
            TrainMean("train_mean"),
            TreeStructured("hgb_structured", "hgb", False),
            TreeStructured("rf_structured", "rf", False),
            TreeStructured("et_structured", "et", False),
            TreeStructured("et_structured_text", "et", True),
            Ensemble("ens_et_hgb_ridge", [TreeStructured("a", "et", True), TreeStructured("b", "hgb", False), RidgeText("c", 3.0, True)]),
            SeedEnsemble("ens_struct_seeds", False),
            SeedEnsemble("ens_struct_seeds_text", True),
        ]
    return [
        TrainMean("train_mean"),
        TreeStructured("et_structured", "et", False),
        TreeStructured("rf_structured", "rf", False),
        TreeStructured("hgb_structured", "hgb", False),
        TreeStructured("et_structured_text", "et", True),
        TreeStructured("hgb_structured_text", "hgb", True),
        RidgeText("ridge_tfidf", 1.0, False),
        RidgeText("ridge_tfidf_structured", 3.0, True),
        KNNText("knn5_tfidf", 5),
        Ensemble("ens_et_hgb_ridge", [TreeStructured("a", "et", True), TreeStructured("b", "hgb", False), RidgeText("c", 3.0, True)]),
        PairwiseRanker("pair_hgb_sibling", "hgb", False, False),
        PairwiseRanker("pair_et_sibling", "et", False, False),
        PairwiseRanker("pair_hgb_withinrun", "hgb", False, True),
        PairwiseRanker("pair_hgb_sibling_text", "hgb", True, False),
    ]


# ----------------------------------------------------------------------------- evaluation


def decision_sets(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cell_id"], tuple(r["parents"]))].append(r)
    out = []
    for key, g in sorted(groups.items()):
        if len(g) >= 2:
            g = sorted(g, key=lambda r: r["plan_at"])
            out.append({"cell_id": key[0], "parents": list(key[1]), "cards": g})
    return out


def selection_metrics(rows, pred, min_gap=0.0):
    """Sibling-set metrics for a scoring function pred[example_id] -> score."""
    sets = decision_sets(rows)
    pair_hits = pair_total = 0.0
    top1 = regret = chosen = best = worst = mean_ = later = 0.0
    n = 0
    for s in sets:
        ys = [c["official_accuracy"] for c in s["cards"]]
        if max(ys) - min(ys) < min_gap:
            continue
        n += 1
        ps = [pred[c["example_id"]] for c in s["cards"]]
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                if abs(ys[i] - ys[j]) < 0.01:
                    continue
                pair_total += 1
                if ps[i] == ps[j]:
                    pair_hits += 0.5
                elif (ps[i] > ps[j]) == (ys[i] > ys[j]):
                    pair_hits += 1
        k = int(np.argmax(ps))
        chosen += ys[k]
        best += max(ys)
        worst += min(ys)
        mean_ += float(np.mean(ys))
        later += ys[-1]
        top1 += float(ys[k] >= max(ys) - 1e-12)
        regret += max(ys) - ys[k]
    if n == 0:
        return {}
    return {
        "n_sets": n,
        "pairwise_acc": pair_hits / max(pair_total, 1),
        "n_pairs": int(pair_total),
        "top1_acc": top1 / n,
        "regret": regret / n,
        "chosen_acc": chosen / n,
        "oracle_acc": best / n,
        "random_acc": mean_ / n,
        "later_sibling_acc": later / n,
        "worst_acc": worst / n,
    }


def point_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "spearman": float(spearmanr(y, p).correlation) if len(y) > 2 else float("nan"),
        "pearson": float(pearsonr(y, p)[0]) if len(y) > 2 and np.std(p) > 0 else float("nan"),
    }


EXTRA_ROWS = []  # training-only rows (label in "official_accuracy" already set by loader)


def load_extra(path, mode):
    """Corpus rows labeled with local accuracy; used for training folds only."""
    if not path:
        return []
    out = []
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        if r.get("local_accuracy") is None or not r.get("lineage_complete"):
            continue
        if mode == "gemma" and "gemma" not in str(r.get("base_model", "")).lower():
            continue
        r = dict(r)
        r["official_accuracy"] = r["local_accuracy"]
        r["is_extra"] = True
        out.append(r)
    return out


def cross_validate(rows, fold_of, variants, inner_select=True, log=print):
    y_all = {r["example_id"]: r["official_accuracy"] for r in rows}
    preds = {v.name: {} for v in variants}
    preds["nested"] = {}
    nested_choice = {}
    folds = sorted(set(fold_of.values()))
    for fold in folds:
        train = [r for r in rows if fold_of[r["cell_id"]] != fold] + EXTRA_ROWS
        test = [r for r in rows if fold_of[r["cell_id"]] == fold]
        if not test:
            continue
        ytr = np.array([r["official_accuracy"] for r in train])
        for v in variants:
            v.fit(train, ytr)
            for r, p in zip(test, v.predict(test)):
                preds[v.name][r["example_id"]] = float(p)
        if inner_select:
            # choose the variant by inner run-held-out pairwise accuracy on the training runs
            core = [r for r in train if not r.get("is_extra")]
            inner_fold = make_folds(core, k=4, seed=fold + 100)
            inner_scores = {}
            for v in variants:
                if v.name == "train_mean":
                    continue
                ip = {}
                for ifold in sorted(set(inner_fold.values())):
                    itr = [r for r in core if inner_fold[r["cell_id"]] != ifold] + EXTRA_ROWS
                    ite = [r for r in core if inner_fold[r["cell_id"]] == ifold]
                    v.fit(itr, np.array([r["official_accuracy"] for r in itr]))
                    for r, p in zip(ite, v.predict(ite)):
                        ip[r["example_id"]] = float(p)
                sm = selection_metrics(core, ip)
                pm = point_metrics([y_all[k] for k in ip], [ip[k] for k in ip])
                inner_scores[v.name] = sm.get("pairwise_acc", 0) - 0.5 * pm["mae"]
            best = max(inner_scores, key=inner_scores.get)
            nested_choice[fold] = best
            for r in test:
                preds["nested"][r["example_id"]] = preds[best][r["example_id"]]
            log(f"fold {fold}: test runs={len({r['cell_id'] for r in test})} inner pick={best} ({inner_scores[best]:.3f})")
    results = {}
    for name, p in preds.items():
        if not p:
            continue
        ids = [r["example_id"] for r in rows if r["example_id"] in p]
        results[name] = {
            "point": point_metrics([y_all[i] for i in ids], [p[i] for i in ids]),
            "selection": selection_metrics([r for r in rows if r["example_id"] in p], p),
            "selection_gap2": selection_metrics([r for r in rows if r["example_id"] in p], p, min_gap=0.02),
        }
    return preds, results, nested_choice


def baseline_selection(rows):
    """Reference selectors that need no model."""
    out = {}
    later = {r["example_id"]: float(datetime_key(r["plan_at"])) for r in rows}
    out["later_sibling"] = selection_metrics(rows, later)
    out["earlier_sibling"] = selection_metrics(rows, {k: -v for k, v in later.items()})
    rng = random.Random(0)
    rand = [selection_metrics(rows, {r["example_id"]: rng.random() for r in rows}) for _ in range(200)]
    out["random"] = {k: float(np.mean([x[k] for x in rand])) for k in rand[0]}
    return out


def datetime_key(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def fmt_table(results, baselines):
    lines = ["| variant | MAE | Spearman | pairwise acc | top-1 | regret | chosen acc | n sets |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        s = r["selection"]
        lines.append(f"| {name} | {r['point']['mae']:.3f} | {r['point']['spearman']:.3f} | {s['pairwise_acc']:.3f} | {s['top1_acc']:.3f} | {s['regret']:.3f} | {s['chosen_acc']:.3f} | {s['n_sets']} |")
    for name, s in baselines.items():
        lines.append(f"| baseline: {name} | | | {s['pairwise_acc']:.3f} | {s['top1_acc']:.3f} | {s['regret']:.3f} | {s['chosen_acc']:.3f} | {s['n_sets']} |")
    s = next(iter(results.values()))["selection"]
    lines.append(f"| oracle | | | 1.000 | 1.000 | 0.000 | {s['oracle_acc']:.3f} | {s['n_sets']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=Path, default=Path("data/analysis/wm_study/cards.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/analysis/wm_study/wm_cv"))
    ap.add_argument("--benchmarks", nargs="*", default=["gsm8k", "aime2025"])
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--extra-cards", type=Path, default=None)
    ap.add_argument("--extra-mode", default="gemma", choices=["gemma", "all"])
    ap.add_argument("--extra-weight", type=float, default=1.0)
    args = ap.parse_args()
    global EXTRA_ROWS
    EXTRA_ROWS = load_extra(args.extra_cards, args.extra_mode)
    for r in EXTRA_ROWS:
        r["weight_scale"] = args.extra_weight
    if EXTRA_ROWS:
        print(f"extra training rows: {len(EXTRA_ROWS)} from {len({r['cell_id'] for r in EXTRA_ROWS})} corpus runs (mode={args.extra_mode}, weight x{args.extra_weight})")
    for benchmark in args.benchmarks:
        rows = [r for r in load_cards(args.cards, benchmark) if r["lineage_complete"]]
        fold_of = make_folds(rows, args.folds)
        print(f"== {benchmark}: {len(rows)} labeled cards, {len(fold_of)} runs, {args.folds} folds")
        preds, results, nested = cross_validate(rows, fold_of, make_variants())
        baselines = baseline_selection(rows)
        out = args.out / benchmark
        out.mkdir(parents=True, exist_ok=True)
        (out / "folds.json").write_text(json.dumps(fold_of, indent=1, sort_keys=True))
        (out / "metrics.json").write_text(json.dumps({"results": results, "baselines": baselines, "nested_choice": nested}, indent=1))
        with (out / "predictions.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps({"example_id": r["example_id"], "cell_id": r["cell_id"], "fold": fold_of[r["cell_id"]], "y": r["official_accuracy"], **{k: v.get(r["example_id"]) for k, v in preds.items()}}) + "\n")
        table = fmt_table(results, baselines)
        (out / "table.md").write_text(table + "\n")
        print(table)


if __name__ == "__main__":
    main()
