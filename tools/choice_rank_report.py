"""Top-3 regret over a real choice set, against the baseline that saturates it.

`regret@k = best_accuracy_in_the_set - best_accuracy_among_the_k_you_picked`.
Zero means you picked the winner. This is the thesis metric, and the reason it
needs care is `a-lookup-table-saturates-the-gsm8k-gemma-split`: a table of
"which agent scores well on average" reaches regret 0.0000 on a full cell,
because a cell holds ~40 runs from ~20 agent families and agent identity is 66 %
of the variance. Any arm that looks good on a full cell has to be shown again
somewhere that table cannot work.

So everything is reported on three populations:

  full cell          ~40 candidates, agent identity visible and decisive
  within family      the sub-set of a cell written by one agent family
  agent-blind cell   a full cell, but every arm that could use agent identity
                     is replaced by its leave-one-out-by-family version

and against four baselines: random, the agent-family lookup table, the 21
bucketed features, and the self-report regex. `--bootstrap` resamples cells,
since 28 cells is the real sample size, not 1,175 runs.

Every arm is a SCORE, and every score is reported twice: once with ties settled
by job id, which is what the pipeline does, and once with ties settled at random.
An arm that is flat where the top-3 cut falls is not ranking -- the job id is,
and job ids are issued in time order, so that arm is really "prefer whatever ran
first". The agent-family table is constant inside a family and its within-family
row is exactly that: 0.0546 by job id against 0.0247 under random tie-breaks,
worse than every one of 200 draws. Read that row as an artefact, not a result.

`random` is an expectation over draws, never an outcome, so its solve rate is the
exact `1 - C(n-m, k)/C(n, k)` and not a `regret == 0` test on the expectation --
that test counts sets where every k-subset happens to win, which is a different
quantity and reads far too low (0.0 % against the true 11.4 %).

    python3 tools/choice_rank_report.py
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools/splitdx")]

import choice_rank as CR  # noqa: E402
import recipe_signal2 as RS  # noqa: E402
import traj_read as T  # noqa: E402

K = 3


def regret(ranked, k=K):
    """ranked: rows in the arm's preferred order. Uses true accuracy only to score."""
    if not ranked:
        return float("nan")
    best = max(r["accuracy"] for r in ranked)
    return best - max(r["accuracy"] for r in ranked[:k])


def rnd_regret(rows, k=K, n=2000, seed=0):
    """Expected regret of picking k of the set uniformly at random, by simulation.

    This is an EXPECTATION over draws, not the outcome of one draw, so it must
    never be fed to a `regret == 0` test the way a real arm's per-set regret is:
    that counts sets where every k-subset wins, not sets this arm solves. Use
    `rnd_solved` for the solve rate.
    """
    a = np.array([r["accuracy"] for r in rows])
    if len(a) <= k:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n, len(a))), axis=1)[:, :k]
    return float(np.mean(a.max() - a[idx].max(axis=1)))


def rnd_solved(rows, k=K):
    """P(a uniform k-subset contains a set maximum) = 1 - C(n-m, k)/C(n, k),
    with m the number of runs tied at the maximum. Exact, no simulation."""
    acc = [r["accuracy"] for r in rows]
    n = len(acc)
    if n <= k:
        return 1.0
    best = max(acc)
    m = sum(1 for x in acc if x >= best - 1e-12)
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


# --- the arms, each a function rows -> one score per row, higher is better -----
#: Arms are SCORES rather than orders so that `_order` can settle ties two ways:
#: by job id, which is what the pipeline does, or at random, which is what the
#: tie-break column reports. This matters because several arms are flat over
#: large blocks -- the agent-family table is literally constant inside a family,
#: so on that population its job-id order, not the table, decides the metric.

def _order(rows, score, rng=None):
    rows = list(rows)
    score = np.asarray(score, dtype=float)
    if rng is None:
        keys = [(-score[i], 0.0, r["run"]) for i, r in enumerate(rows)]
    else:
        j = rng.random(len(rows))
        keys = [(-score[i], float(j[i]), r["run"]) for i, r in enumerate(rows)]
    return [r for _, r in sorted(zip(keys, rows), key=lambda t: t[0])]


def sc_selfreport(rows, sr):
    return [sr[r["run"]] if sr.get(r["run"]) is not None else -1.0 for r in rows]


def sc_famtable(rows, fam_mean):
    """The table that saturates. Leave-one-out by construction: `fam_mean` is
    built from every OTHER cell, so it never sees this cell's own accuracies."""
    return [fam_mean.get(r["fam"], 0.0) for r in rows]


def sc_stage_a(rows, a):
    """z(quality) + z(predicted_accuracy). See CR.rank_key for why both fields."""
    return CR.rank_score(rows, a)


def sc_features(rows, model, vec):
    if model is None:
        return np.zeros(len(rows))
    return model.decision_function(np.array([vec(r) for r in rows]))


def sc_stage_ab(cell, rows, short, b, a):
    """Copeland over the shortlist, stage A below it, as one score. `wins` is an
    integer and the stage-A term is far under 100, so this is exactly the
    lexicographic order `CR.copeland` produces, only tie-breakable."""
    sa = dict(zip([r["run"] for r in rows], CR.rank_score(rows, a)))
    wins = CR.copeland_wins(cell, short, b)
    keep = {r["run"] for r in short}
    return [(1e6 if r["run"] in keep else 0.0)
            + 100.0 * wins[r["run"]] + sa[r["run"]] for r in rows]


_FIT_CACHE: dict = {}


def fit_features(rows, cells_, held_cell):
    """Pairwise logistic on feature differences, trained on every other cell.
    Held out by CELL, not by row: a within-cell ranking model trained on rows
    from the same cell would be scoring its own training data."""
    if held_cell in _FIT_CACHE:
        return _FIT_CACHE[held_cell]
    import warnings

    from scipy.optimize import OptimizeWarning
    from sklearn.linear_model import LogisticRegression

    warnings.simplefilter("ignore", OptimizeWarning)
    feats = [f for f in RS.feat(rows[0]) if f not in RS.SKIP3]
    levels = {f: sorted({r[f] for r in rows}) for f in feats}
    off, o = {}, 0
    for f in feats:
        off[f] = o
        o += len(levels[f])
    index = {f: {v: i for i, v in enumerate(levels[f])} for f in feats}

    def vec(r):
        v = np.zeros(o)
        for f in feats:
            v[off[f] + index[f][r[f]]] = 1.0
        return v

    X, y = [], []
    for c, rs in cells_.items():
        if c == held_cell:
            continue
        for i, x in enumerate(rs):
            for z in rs[i + 1:]:
                if abs(x["accuracy"] - z["accuracy"]) < 0.02:
                    continue
                X.append(vec(x) - vec(z))
                y.append(int(x["accuracy"] > z["accuracy"]))
    if len(set(y)) < 2:
        _FIT_CACHE[held_cell] = (None, vec)
        return _FIT_CACHE[held_cell]
    X = np.array(X)
    y = np.array(y)
    m = LogisticRegression(max_iter=3000, C=0.1, fit_intercept=False)
    m.fit(np.vstack([X, -X]), np.concatenate([y, 1 - y]))
    _FIT_CACHE[held_cell] = (m, vec)
    return m, vec


def main() -> int:
    rows = T.load_rows()
    cells_ = CR.cells(rows)

    a = {}
    p = CR.OUT / "score_a.jsonl"
    for line in p.open():
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        a[d["run"]] = d
    have = sum(1 for r in rows if isinstance(a.get(r["run"], {}).get("quality"),
                                             (int, float)))
    print(f"{len(rows)} runs, {len(cells_)} cells; stage A scored {have}")

    b = {}
    pb = CR.OUT / "score_b.jsonl"
    if pb.exists():
        for line in pb.open():
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            b[(d["cell"], d["x"], d["y"])] = d.get("winner")
    print(f"stage B: {len(b)} ordered comparisons\n")

    # self-report maxima, reused from traj_read
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(32) as ex:
        sr = {run: v for run, v, _ in
              ex.map(T._sr_worker, [(r["experiment"], r["run"], False)
                                    for r in rows], chunksize=8)}

    # --- 0. sanity: does stage A rank at all? --------------------------------
    from scipy.stats import spearmanr
    rho = []
    for c, rs in cells_.items():
        q = [a.get(r["run"], {}).get("quality") for r in rs]
        ok = [(x, r["accuracy"]) for x, r in zip(q, rs)
              if isinstance(x, (int, float))]
        if len(ok) > 5 and len({x for x, _ in ok}) > 1:
            rho.append(spearmanr([x for x, _ in ok], [y for _, y in ok]).statistic)
    print(f"stage A within-cell Spearman: median {np.median(rho):+.3f}, "
          f"range {min(rho):+.3f}..{max(rho):+.3f} over {len(rho)} cells")
    pa = [a.get(r["run"], {}).get("predicted_accuracy") for r in rows]
    ok = [(x, r["accuracy"]) for x, r in zip(pa, rows)
          if isinstance(x, (int, float))]
    err = np.mean([abs(x - y) for x, y in ok])
    print(f"  its absolute accuracy guess is off by {err:.3f} on average "
          f"(n={len(ok)}) -- ranking is the usable part, not calibration\n")

    # --- the three populations ------------------------------------------------
    pops = {
        "full cell": [(c, rs) for c, rs in cells_.items()],
        "within family": [],
    }
    fg = collections.defaultdict(list)
    for r in rows:
        fg[(r["benchmark"], r["trained_model"], r["fam"])].append(r)
    pops["within family"] = [(c, sorted(v, key=lambda r: r["run"]))
                             for c, v in sorted(fg.items()) if len(v) >= 4]

    ARMS = ("random", "agent table", "21 features", "self-report",
            "stage A", "stage A+B")
    keep_full = {}
    for popname, sets in pops.items():
        spread = [max(r["accuracy"] for r in rs) - min(r["accuracy"] for r in rs)
                  for _, rs in sets]
        print(f"=== {popname}: {len(sets)} sets, median size "
              f"{int(np.median([len(rs) for _, rs in sets]))}, "
              f"median spread {np.median(spread):.3f} ===")

        res = collections.defaultdict(list)
        res1 = collections.defaultdict(list)
        sol = collections.defaultdict(list)
        scores = []          # per set: {arm: score vector}, for the tie-break pass
        bcov = []
        for c, rs in sets:
            cell = c[:2]
            # leave-this-cell-out family table
            fam_acc = collections.defaultdict(list)
            for r in rows:
                if (r["benchmark"], r["trained_model"]) != cell:
                    fam_acc[r["fam"]].append(r["accuracy"])
            fam_mean = {k: float(np.mean(v)) for k, v in fam_acc.items()}
            model, vec = fit_features(rows, cells_, cell)

            short = CR.rank_key(rs, a)[:6]
            key = f"{cell[0]}|{cell[1]}"
            got = sum(1 for x, y in itertools.permutations(short, 2)
                      if (key, x["run"], y["run"]) in b)
            n = len(short)
            bcov.append(got / max(1, n * (n - 1)))

            sc = {
                "agent table": sc_famtable(rs, fam_mean),
                "21 features": sc_features(rs, model, vec),
                "self-report": sc_selfreport(rs, sr),
                "stage A": sc_stage_a(rs, a),
                "stage A+B": sc_stage_ab(cell, rs, short, b, a),
            }
            scores.append(sc)
            # random is an expectation, not a draw, so it gets its own two exact
            # numbers and never goes through `regret`.
            res["random"].append(rnd_regret(rs, 3))
            res1["random"].append(rnd_regret(rs, 1))
            sol["random"].append(rnd_solved(rs, 3))
            for name, s in sc.items():
                o = _order(rs, s)
                res[name].append(regret(o, 3))
                res1[name].append(regret(o, 1))
                sol[name].append(float(regret(o, 3) < 1e-9))

        # --- how much of each arm is the job-id tie-break? --------------------
        #: An arm that is flat where the cut falls is not choosing; `r["run"]` is,
        #: and job ids are issued in time order, so that is "prefer whatever ran
        #: first". Re-score every arm under random tie-breaks and print where the
        #: shipped job-id order sits in that distribution.
        tb = {}
        for name in ARMS[1:]:
            sims = []
            for s in range(200):
                rg = np.random.default_rng(7000 + s)
                sims.append(np.mean([regret(_order(rs, scores[i][name], rg))
                                     for i, (_, rs) in enumerate(sets)]))
            tb[name] = np.array(sims)

        print(f'{"arm":>14} {"regret@3":>9} {"95% CI":>16} '
              f'{"solved@3":>9} {"regret@1":>9} {"tie-break":>19}')
        rng = np.random.default_rng(0)
        for name in ARMS:
            v = np.array(res[name])
            bs = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
            if name in tb:
                t = tb[name]
                pct = float((t < v.mean()).mean())
                note = (f"{t.mean():.4f} p{pct * 100:3.0f}" if t.std() > 1e-9
                        else "   none")
            else:
                note = "     n/a"
            print(f"{name:>14} {v.mean():9.4f} "
                  f"[{np.quantile(bs,0.025):6.4f},{np.quantile(bs,0.975):6.4f}] "
                  f"{np.mean(sol[name]):8.1%} {np.mean(res1[name]):9.4f} "
                  f"{note:>19}")
        print("  tie-break = mean regret@3 under RANDOM tie-breaks, and the "
              "percentile the")
        print("  shipped job-id order sits at within that distribution; p100 "
              "means the job-id")
        print("  order was worse than every random one, i.e. the row is an "
              "artefact of it.")
        print(f"  stage-B pair coverage of each set's own shortlist: "
              f"{np.mean(bcov):.1%}")

        # paired: 28 sets is the sample size, and the arms see the same sets
        print(f'\n  paired vs the arm above it ({len(sets)} sets, '
              f'bootstrap over sets + sign test on non-ties)')
        from scipy.stats import binomtest

        def paired(lo, hi):
            d = np.array(res[lo]) - np.array(res[hi])   # >0 means `hi` is better
            bs = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
            w = int((d > 1e-9).sum())
            ls = int((d < -1e-9).sum())
            pv = binomtest(w, w + ls).pvalue if w + ls else 1.0
            print(f"    {hi:>12} - {lo:<12} {-d.mean():+8.4f} "
                  f"[{-np.quantile(bs,0.975):+7.4f},{-np.quantile(bs,0.025):+7.4f}] "
                  f"  {w}-{ls}  p={pv:.3g}")

        for lo, hi in zip(ARMS, ARMS[1:]):
            paired(lo, hi)
        for ref in ("agent table", "random"):
            print(f"  and against {ref}:")
            for hi in ARMS:
                if hi != ref:
                    paired(ref, hi)
        print()
        if popname == "full cell":
            keep_full = {"sets": sets, "res": dict(res)}

    # --- where the agent table cannot help ------------------------------------
    print("=== the same question with the winner's family made ambiguous ===")
    print("    (full cells, restricted to the runs whose family also produced "
          "a below-median run in that cell)")
    res = collections.defaultdict(list)
    sol = collections.defaultdict(list)
    for c, rs in cells_.items():
        med = np.median([r["accuracy"] for r in rs])
        byfam = collections.defaultdict(list)
        for r in rs:
            byfam[r["fam"]].append(r)
        keep = [r for f, v in byfam.items() if len(v) > 1
                and min(x["accuracy"] for x in v) < med
                and max(x["accuracy"] for x in v) > med for r in v]
        if len(keep) < 6:
            continue
        fam_acc = collections.defaultdict(list)
        for r in rows:
            if (r["benchmark"], r["trained_model"]) != c:
                fam_acc[r["fam"]].append(r["accuracy"])
        fam_mean = {k: float(np.mean(v)) for k, v in fam_acc.items()}
        model, vec = fit_features(rows, cells_, c)
        res["random"].append(rnd_regret(keep))
        sol["random"].append(rnd_solved(keep))
        for name, s in (("agent table", sc_famtable(keep, fam_mean)),
                        ("21 features", sc_features(keep, model, vec)),
                        ("self-report", sc_selfreport(keep, sr)),
                        ("stage A", sc_stage_a(keep, a))):
            g = regret(_order(keep, s))
            res[name].append(g)
            sol[name].append(float(g < 1e-9))
    n = len(res["random"])
    print(f'    {n} cells qualify\n{"arm":>14} {"regret@3":>9} {"solved":>8}')
    for name in ("random", "agent table", "21 features", "self-report", "stage A"):
        print(f"{name:>14} {np.mean(res[name]):9.4f} {np.mean(sol[name]):7.1%}")

    # --- leak control ---------------------------------------------------------
    #: Stage A reads the REDACTED digest, so the question is what is left to read
    #: in that text. Answer it by running the score regex on exactly the string
    #: stage A was given: whatever margin stage A holds over THAT is not
    #: transcription of a printed score. (Conditioning on "no run printed a
    #: number" is the obvious cut and it is useless here -- 1 within-family set
    #: out of 76 qualifies.)
    print("\n=== leak control: the regex re-run on the text stage A actually saw ===")
    with ProcessPoolExecutor(32) as ex:
        srr = {run: v for run, v, _ in
               ex.map(T._sr_worker, [(r["experiment"], r["run"], True)
                                     for r in rows], chunksize=8)}
    nraw = sum(1 for r in rows if sr.get(r["run"]) is not None)
    nred = sum(1 for r in rows if srr.get(r["run"]) is not None)
    print(f"  runs with a quotable score: {nraw}/{len(rows)} raw, "
          f"{nred} after redaction")
    for popname, sets in pops.items():
        d = collections.defaultdict(list)
        for c, rs in sets:
            d["self-report, raw text"].append(
                regret(_order(rs, sc_selfreport(rs, sr))))
            d["self-report, redacted"].append(
                regret(_order(rs, sc_selfreport(rs, srr))))
            d["stage A (redacted)"].append(regret(_order(rs, sc_stage_a(rs, a))))
            d["random"].append(rnd_regret(rs))
        print(f"  {popname} ({len(sets)} sets)")
        for k, v in d.items():
            print(f"    {k:>22} {np.mean(v):7.4f}")

    # --- per cell, so a mean is never the only thing on offer -----------------
    print("\n=== every cell, full-cell population ===")
    print(f'{"benchmark":>22} {"base model":>26} {"n":>4} {"best":>6} '
          f'{"spread":>7} {"table":>7} {"A":>7} {"A+B":>7}')
    sets = keep_full["sets"]
    res = keep_full["res"]
    order = np.argsort([-(max(r["accuracy"] for r in rs)
                          - min(r["accuracy"] for r in rs))
                        for _, rs in sets])
    for i in order:
        c, rs = sets[i]
        accs = [r["accuracy"] for r in rs]
        print(f"{c[0][:22]:>22} {c[1].split('/')[-1][:26]:>26} {len(rs):4d} "
              f"{max(accs):6.3f} {max(accs)-min(accs):7.3f} "
              f'{res["agent table"][i]:7.4f} {res["stage A"][i]:7.4f} '
              f'{res["stage A+B"][i]:7.4f}')
    tight = [i for i in range(len(sets))
             if (lambda a: max(a) - min(a))([r["accuracy"] for r in sets[i][1]]) < 0.1]
    if tight:
        keepi = [i for i in range(len(sets)) if i not in tight]
        print(f"  {len(tight)} cells have a spread below 0.10 -- there is almost "
              f"nothing to win or lose in them; excluding them, stage A is "
              f'{np.mean([res["stage A"][i] for i in keepi]):.4f}, stage A+B is '
              f'{np.mean([res["stage A+B"][i] for i in keepi]):.4f} '
              f'and the agent table is '
              f'{np.mean([res["agent table"][i] for i in keepi]):.4f}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
