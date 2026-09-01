"""Top-3 regret over a real choice set, against the baseline that saturates it.

`regret@k = best_accuracy_in_the_set - best_accuracy_among_the_k_you_picked`.
Zero means you picked the winner. This is the thesis metric, and the reason it
needs care is `a-lookup-table-saturates-the-gsm8k-gemma-split`: a table of
"which agent scores well on average" reaches regret 0.0000 on a full cell,
because a cell holds ~40 runs from ~20 agent families and agent identity is 66 %
of the variance. Any arm that looks good on a full cell has to be shown again
somewhere that table cannot work.

So everything is reported on four populations:

  full cell          ~40 candidates, agent identity visible and decisive
  within family      the sub-set of a cell written by one agent family
  within scaffold    the sub-set of a cell produced by one harness
                     (claude_code / codex / opencode / cursor)
  agent-blind cell   a full cell, but every arm that could use agent identity
                     is replaced by its leave-one-out-by-family version

and against four baselines: random, the agent-family lookup table, the 21
bucketed features, and the self-report regex. `--bootstrap` resamples cells,
since 28 cells is the real sample size, not 1,175 runs.

`within scaffold` is here because `within family` wins partly by blinding the
opponent: the family table is CONSTANT on 100 % of within-family sets and on 0 %
of within-scaffold ones, which hold a median of 5 distinct families and never
fewer than 3. It is also a condition you hold before any run exists -- which
harness you operate -- rather than a property of what the agent turned out to do,
so it is a choice set someone is actually in. The header prints that blinding
fraction for every population, so a row that wins by muting the baseline cannot
be read as a result.

Sub-sets of one cell are NOT independent draws: they share that cell's runs, its
stage-A calls and its leave-one-out family table, and their per-cell advantages
correlate +0.54 (scaffold) to +0.79 (peft) with the full cell's. So every paired
test below is CLUSTERED on the parent cell -- the bootstrap resamples the 28
cells and the sign test counts cells, not sets. The set-level counts are printed
alongside and are systematically more significant; that gap is the over-count,
not evidence. Calibration: shuffling accuracies within each cell rejects the
percentile CI 8-10 % of the time against a nominal 5 % on every population,
while the clustered sign test rejects 3-4 %, so the sign test is the load-bearing
one and the CI should be read as ~90 %. And cutting each cell into RANDOM chunks
of the within-scaffold size profile gives +0.0094 [+0.0034,+0.0169] over 200
draws against scaffold's +0.0122 at the 19th percentile -- the extra power comes
from measuring the same 28 cells on smaller sets, and scaffold is a typical such
partition rather than a lucky one.

A baseline has to be given the scale it is averaged on. The agent-family table
used to average RAW accuracies across cells, which understated it badly: cell
mean accuracy runs 0.003 to 0.820, so the table was mostly reporting which
family drew the easy cells. Averaging within-cell z-scores instead takes it from
0.0248 to 0.0070 on full cells, better in 11 of 28 cells and worse in none, and
that DELETES the headline -- `stage A - agent table` was -0.0167, 10-2,
p=0.0386, and is now +0.0011, 3-6, p=0.508, i.e. at k=3 on a full cell stage A
and the lookup table are indistinguishable. That tie is specific to k=3 and to
the whole cell: at k=1 the same full cells give stage A 0.0200 against the
corrected table's 0.0480 (stage A better in 200/200 random tie-breaks), and on
the ambiguous-family cells below it is 0.0135 against 0.0265, -0.0162 under
matched random tie-breaks, 12-1, sign p=0.0034 against an MDE of 0.0133. The
within-family population, where the table cannot work at all, is the largest
but not the only thing left. See `cell_z`.

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


def cell_z(rows):
    """Every run's accuracy re-expressed as a z-score WITHIN its own cell.

    The family lookup table is an average over cells, and a raw accuracy is not
    comparable across cells: cell mean accuracy runs 0.003 to 0.820 and the
    median within-cell spread is 0.655, so a family that happened to be given
    the easy cells scores high for a reason that is not the family. It is not a
    hypothetical -- `claude-opus-5` ran in 24 of 28 cells with mean difficulty
    0.291 against `claude-opus-4-8`'s 0.334, so the raw mean ranks it THIRD
    while every scale-corrected table ranks it second, and it is the family that
    produced the winning run in most cells. Correlation between a family's raw
    mean and the mean accuracy of the cells it was run in is -0.171; on z-scores
    it is -0.508, i.e. the strong families were deliberately given the hard cells.
    """
    acc = collections.defaultdict(list)
    for r in rows:
        acc[(r["benchmark"], r["trained_model"])].append(r)
    out = {}
    for rs in acc.values():
        v = np.array([r["accuracy"] for r in rs], dtype=float)
        s = v.std()
        z = (v - v.mean()) / (s if s > 1e-12 else 1.0)
        out.update({r["run"]: float(z[i]) for i, r in enumerate(rs)})
    return out


def fam_table(rows, zacc, held_cell):
    """Leave-one-cell-out family table, on the within-cell scale.

    Built from every OTHER cell, so it never sees this cell's own accuracies.
    Mean-of-z (this), median-of-z, mean-of-within-cell-percentile, median-of-
    percentile, mean of the within-cell min-max score and the MEDIAN of the raw
    accuracies all give the SAME 28 per-cell regrets, 0.0070; only the raw MEAN
    is different, at 0.0248. So this is a correction, not a choice of estimator.
    """
    acc = collections.defaultdict(list)
    for r in rows:
        if (r["benchmark"], r["trained_model"]) != held_cell:
            acc[r["fam"]].append(zacc[r["run"]])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def sc_famtable(rows, fam_mean):
    """The table that saturates, scored on the scale it is averaged over.

    The default is 0.0, which on the z scale is an average family rather than
    the worst one; 1 of 1175 runs has a family that appears in no other cell and
    the number is the same under either default.
    """
    return [fam_mean.get(r["fam"], 0.0) for r in rows]


def sc_stage_a(rows, a):
    """z(quality) + z(predicted_accuracy). See CR.rank_key for why both fields."""
    return CR.rank_score(rows, a)


def sc_features(rows, model, vec):
    if model is None:
        return np.zeros(len(rows))
    return model.decision_function(np.array([vec(r) for r in rows]))


def sc_stage_ab(cell, rows, short, b, a):
    """Copeland over the shortlist, stage A below it, as one score.

    The Copeland term is a win RATE and not a win count -- see
    `CR.copeland_score`. With the cached pairs covering 37.6 % of this
    shortlist, a count ranks the 38.7 % of shortlisted runs that were never
    compared dead last on no evidence, which is what the shipped +0.0026
    "stage B costs points" was. It is also built from POSITIONS rather than by
    scaling two numbers together, so it is exactly the lexicographic order
    (Copeland, then stage A) whatever the two terms' magnitudes turn out to be:
    the old form assumed the stage-A term was "far under 100" and it can reach
    8.2, which is fine against integer wins and would not be against a rate.
    """
    sa = dict(zip([r["run"] for r in rows], CR.rank_score(rows, a)))
    cs = CR.copeland_score(cell, short, b)
    pos = {r["run"]: i for i, r in enumerate(
        sorted(short, key=lambda r: (-cs[r["run"]], -sa[r["run"]])))}
    return [1e6 - pos[r["run"]] if r["run"] in pos else sa[r["run"]]
            for r in rows]


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
    zacc = cell_z(rows)

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
          f"(n={len(ok)}) -- ranking is the usable part, not calibration")

    #: Both stages judged on the SAME pairs, split by how far apart the two runs
    #: really scored. This is the diagnostic that says where the ceiling is, and
    #: it is the reason `sc_stage_ab` lets Copeland override stage A rather than
    #: blending them: stage B is the better judge, but only outside the narrow
    #: band, and the narrow band is exactly the top of a shortlist. Comparing
    #: the two on DIFFERENT pair sets says the opposite and is how this was
    #: first read backwards -- stage A looked like the stronger judge because it
    #: was being graded on the easier 772.
    acc = {r["run"]: r["accuracy"] for r in rows}
    sa_all = {}
    for c, rs in cells_.items():
        s = CR.rank_score(rs, a)
        sa_all.update({(f"{c[0]}|{c[1]}", r["run"]): s[i]
                       for i, r in enumerate(rs)})
    band = collections.defaultdict(lambda: [0, 0, 0])
    for (c, x, y), w in b.items():
        if w not in ("A", "B") or abs(acc[x] - acc[y]) < 1e-12:
            continue
        g = abs(acc[x] - acc[y])
        k = ("under 0.02" if g < 0.02 else "0.02-0.05" if g < 0.05
             else "0.05-0.15" if g < 0.15 else "over 0.15")
        pb = x if w == "A" else y
        pa_ = x if sa_all[(c, x)] > sa_all[(c, y)] else y
        e = band[k]
        e[2] += 1
        e[0] += acc[pb] > acc[y if pb == x else x]
        e[1] += acc[pa_] > acc[y if pa_ == x else x]
    tt = [sum(band[k][i] for k in band) for i in range(3)]
    print(f"  pairwise accuracy on the {tt[2]} non-tied comparisons stage B was "
          f"asked, by how far apart the runs really scored:")
    print(f"    {'true gap':>12} {'n':>6} {'stage B':>9} {'stage A':>9}")
    for k in ("under 0.02", "0.02-0.05", "0.05-0.15", "over 0.15"):
        o, p, n_ = band[k]
        print(f"    {k:>12} {n_:6d} {100*o/n_:8.1f}% {100*p/n_:8.1f}%")
    print(f"    {'all':>12} {tt[2]:6d} {100*tt[0]/tt[2]:8.1f}% "
          f"{100*tt[1]/tt[2]:8.1f}%\n")

    # --- the three populations ------------------------------------------------
    pops = {
        "full cell": [(c, rs) for c, rs in cells_.items()],
        "within family": [],
    }
    for popname, col in (("within family", "fam"),
                         ("within scaffold", "trace_format")):
        g = collections.defaultdict(list)
        for r in rows:
            g[(r["benchmark"], r["trained_model"], r[col])].append(r)
        pops[popname] = [(c, sorted(v, key=lambda r: r["run"]))
                         for c, v in sorted(g.items()) if len(v) >= 4]

    #: Two shortlist sizes, both reported, because K is a BUDGET and its floor
    #: is knowable before the comparator runs: the oracle over stage A's top K
    #: is 0.0030 at 6 and 0.0018 at 10 on full cells, 0.0017 and 0.0001 within
    #: scaffold. Picking K by which one wins the board would be fitting a
    #: hyperparameter on 28 cells; printing both, next to a cost that grows as
    #: K^2 (30 ordered comparisons per set at 6, 90 at 10), is the honest form.
    ARMS = ("random", "agent table", "21 features", "self-report",
            "stage A", "stage A+B", "stage A+B, K=10")
    keep_full = {}
    for popname, sets in pops.items():
        spread = [max(r["accuracy"] for r in rs) - min(r["accuracy"] for r in rs)
                  for _, rs in sets]
        nfam = [len({r["fam"] for r in rs}) for _, rs in sets]
        print(f"=== {popname}: {len(sets)} sets in "
              f"{len({c[:2] for c, _ in sets})} cells, median size "
              f"{int(np.median([len(rs) for _, rs in sets]))}, "
              f"median spread {np.median(spread):.3f}, family table constant on "
              f"{np.mean([n == 1 for n in nfam]):.0%} of sets ===")

        res = collections.defaultdict(list)
        res1 = collections.defaultdict(list)
        sol = collections.defaultdict(list)
        scores = []          # per set: {arm: score vector}, for the tie-break pass
        bcov, bnil, bshort = [], [], []
        for c, rs in sets:
            cell = c[:2]
            fam_mean = fam_table(rows, zacc, cell)
            model, vec = fit_features(rows, cells_, cell)

            ranked = CR.rank_key(rs, a)
            short, wide = ranked[:6], ranked[:10]
            key = f"{cell[0]}|{cell[1]}"
            got = sum(1 for x, y in itertools.permutations(short, 2)
                      if (key, x["run"], y["run"]) in b)
            n = len(short)
            bcov.append(got / max(1, n * (n - 1)))
            bnil.append(sum(1 for x in short if not any(
                (key, x["run"], y["run"]) in b or (key, y["run"], x["run"]) in b
                for y in short if y is not x)))
            bshort.append(n)

            sc = {
                "agent table": sc_famtable(rs, fam_mean),
                "21 features": sc_features(rs, model, vec),
                "self-report": sc_selfreport(rs, sr),
                "stage A": sc_stage_a(rs, a),
                "stage A+B": sc_stage_ab(cell, rs, short, b, a),
                "stage A+B, K=10": sc_stage_ab(cell, rs, wide, b, a),
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
        #: Keep the PER-SET regrets of every tie-break draw, not just their
        #: mean: the same draws are what says whether a paired row's sign test
        #: survives the tie-break, and that is a different question from
        #: whether either arm's mean does. Both arms of a row see the SAME
        #: jitter on draw s, so the control is paired.
        tbv = {}
        for name in ARMS[1:]:
            sims = []
            for s in range(200):
                rg = np.random.default_rng(7000 + s)
                sims.append([regret(_order(rs, scores[i][name], rg))
                             for i, (_, rs) in enumerate(sets)])
            tbv[name] = np.array(sims)                     # [200, n_sets]
        tb = {k: v.mean(1) for k, v in tbv.items()}

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
              f"{np.mean(bcov):.1%}; {sum(bnil)} of {sum(bshort)} shortlisted "
              f"runs ({sum(bnil) / sum(bshort):.1%}) have NO cached comparison")
        print("  at all, so the stage A+B row is that far from being a "
              "measurement of stage B.")

        #: What the two stages are each still worth, which is not the same
        #: question as which arm wins. `oracle over the top K` is the regret a
        #: PERFECT comparator would reach on stage A's top K, so the gap from
        #: stage A down to it is what any reranker can still buy at that budget
        #: (a PRECISION headroom), and the gap from one K to the next is what
        #: only a wider shortlist can buy (a RECALL headroom). They take
        #: different fixes and the board otherwise shows neither.
        orc = {k: [] for k in (3, 6, 10, 20)}
        for _, rs in sets:
            o = CR.rank_key(rs, a)
            best = max(r["accuracy"] for r in rs)
            for k in orc:
                orc[k].append(best - max(r["accuracy"] for r in o[:k]))
        print("  oracle over stage A's top K, i.e. the floor a perfect "
              "comparator hits at that budget:")
        print("    " + "   ".join(
            f"K={k}: {np.mean(v):.4f} ({np.mean(np.array(v) < 1e-12):.0%} solved)"
            for k, v in orc.items()))

        # paired, CLUSTERED on the parent cell. 28 cells is the sample size on
        # every population here, not the set count: two sub-sets of one cell
        # share that cell's runs, its stage-A calls and its leave-one-out family
        # table, and their per-cell advantages correlate +0.54 to +0.79 with the
        # full cell's, so bootstrapping 84 of them as independent draws
        # over-counts. The set-level counts are printed in the trailing bracket
        # so the size of that over-count is visible rather than assumed away.
        parent = [c[:2] for c, _ in sets]
        cidx: dict = {}
        for i, pc in enumerate(parent):
            cidx.setdefault(pc, []).append(i)
        clus = sorted(cidx)
        nc = len(clus)

        def _clus(v):
            """set-level vector -> per-cell means, so a cell counts once however
            many sub-sets it was cut into. Works on (nsets,) and (ndraws, nsets)."""
            return np.array([np.asarray(v)[..., cidx[c]].mean(axis=-1)
                             for c in clus]).T

        print(f'\n  paired vs the arm above it ({len(sets)} sets in {nc} cells; '
              f'bootstrap and sign test CLUSTERED on the cell; mde = the effect '
              f'this many cells could find at 80 % power, tb = share of the 200 '
              f'random tie-break draws in which the sign test still clears 0.05, '
              f'and the mean effect over those draws)')
        from scipy.stats import binomtest

        def _signp(d):
            w = int((d > 1e-9).sum())
            ls = int((d < -1e-9).sum())
            return w, ls, (binomtest(w, w + ls).pvalue if w + ls else 1.0)

        def paired(lo, hi):
            dset = np.array(res[lo]) - np.array(res[hi])  # >0 means `hi` better
            d = _clus(dset)
            bs = d[rng.integers(0, nc, (4000, nc))].mean(1)
            w, ls, pv = _signp(d)
            #: Two things a bare p hides on 28 sets. First, the smallest effect
            #: this many sets could have found at 80 % power, from the observed
            #: sd of the paired difference: a row whose effect is under its own
            #: MDE is one this design was never powered to detect, whatever p
            #: came out. Second, the sign test is decided by the handful of sets
            #: that differ at all, and which ones those are depends on the
            #: arbitrary job-id tie-break -- so re-run it on the 200 random
            #: tie-break draws and say in how many of them the row still clears
            #: 0.05. A row that clears it by job id and in half the draws is a
            #: property of the job ids.
            sd = d.std(ddof=1)
            #: An all-zero difference has sd 0, so an MDE of 0 would flag a row
            #: with nothing in it as adequately powered. Say n/a instead.
            if sd > 0:
                m = (1.959964 + 0.8416212) * sd / math.sqrt(nc)
                flag = "<MDE" if abs(d.mean()) < m else "    "
                mstr = f"mde={m:.4f} {flag}"
            else:
                mstr = "mde=  n/a     "
            #: tb without its mean effect is the most misleading column here: a
            #: row can clear 0.05 in 99 % of draws at a third of the job-id
            #: effect, which says the effect is real and the headline size is a
            #: property of the job ids. Print both.
            tbp = "    n/a"
            if lo in tbv and hi in tbv:
                dd = _clus(tbv[lo] - tbv[hi])
                ps = [_signp(dd[s])[2] for s in range(dd.shape[0])]
                tbp = f"{np.mean(np.array(ps) < 0.05):4.0%} eff{-dd.mean():+7.4f}"
            sw, sl, spv = _signp(dset)
            tail = f"  (by set {sw}-{sl} p={spv:.3g})" if len(sets) != nc else ""
            print(f"    {hi:>12} - {lo:<12} {-d.mean():+8.4f} "
                  f"[{-np.quantile(bs,0.975):+7.4f},{-np.quantile(bs,0.025):+7.4f}] "
                  f"  {w}-{ls}  p={pv:.3g}  {mstr}  tb{tbp}{tail}")

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
        fam_mean = fam_table(rows, zacc, c)
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

    # --- the run-id ablation --------------------------------------------------
    #: `RC.render` writes `# run: {run}` at the top of the digest, and a PTB run
    #: id ends in the job number, which is issued in TIME order. That is enough
    #: to rank on its own, so "stage A reads the trajectory" is not established
    #: until the id is taken away and the pass is re-run. `--stage noid` does
    #: that -- same prompt, same redaction, job number replaced by an order-free
    #: hash -- over all 28 cells, because restricting it to the cells where the
    #: job-id arm looks strong would select the sample on the statistic at issue.
    print("\n=== run-id ablation: the same pass with the job number removed ===")
    an = {}
    pn = CR.OUT / "score_a_noid.jsonl"
    if pn.exists():
        for line in pn.open():
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            an[d["run"]] = d
    scored = sum(1 for r in rows
                 if isinstance(an.get(r["run"], {}).get("quality"), (int, float)))
    if scored < len(rows):
        print(f"  NOT RUN ({scored}/{len(rows)} scored). "
              f"`python3 tools/choice_rank.py --stage noid`, 1175 calls, ~$302.")
    else:
        from scipy.stats import binomtest

        def jobid(r):
            _, _, j = r["run"].rpartition("_")
            return int(j) if j.isdigit() else 0

        for popname, sets in pops.items():
            cidx = {}
            for i, (c, _) in enumerate(sets):
                cidx.setdefault(c[:2], []).append(i)
            clus = sorted(cidx)
            d = collections.defaultdict(list)
            for c, rs in sets:
                d["random"].append(rnd_regret(rs))
                d["newest three (job id)"].append(
                    regret(sorted(rs, key=lambda r: -jobid(r))))
                d["stage A, id present"].append(regret(_order(rs, sc_stage_a(rs, a))))
                d["stage A, id removed"].append(regret(_order(rs, sc_stage_a(rs, an))))
            print(f"  {popname} ({len(sets)} sets in {len(clus)} cells)")
            for k in ("random", "newest three (job id)",
                      "stage A, id present", "stage A, id removed"):
                print(f"    {k:>22} {np.mean(d[k]):7.4f}")
            #: the number that decides it: how much of stage A's margin over
            #: random survives losing the id, clustered on the cell.
            def cm(v):
                v = np.asarray(v)
                return np.array([v[cidx[c]].mean() for c in clus])
            R_ = cm(d["random"])
            P_ = cm(d["stage A, id present"])
            N_ = cm(d["stage A, id removed"])
            gain_p, gain_n = (R_ - P_).mean(), (R_ - N_).mean()
            keep = gain_n / gain_p if abs(gain_p) > 1e-12 else float("nan")
            bs = []
            nclu = len(clus)
            for _ in range(4000):
                ix = rng.integers(0, nclu, nclu)
                g = (R_[ix] - P_[ix]).mean()
                if abs(g) > 1e-12:
                    bs.append((R_[ix] - N_[ix]).mean() / g)
            lo, hi = np.quantile(bs, [0.025, 0.975]) if bs else (float("nan"),) * 2
            dd = N_ - P_
            w = int((dd > 1e-9).sum())
            ls = int((dd < -1e-9).sum())
            pv = binomtest(w, w + ls).pvalue if w + ls else 1.0
            print(f"    keeps {keep:.0%} of stage A's gain over random "
                  f"[95% CI {lo:.0%}, {hi:.0%}]; removing the id costs "
                  f"{N_.mean() - P_.mean():+.4f}, {ls}-{w} cells, p={pv:.3g}")

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
