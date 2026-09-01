"""Does a model reading the trajectory beat 21 summaries of it?

`recipe_signal2.py` closed the question one level down: of 21 bucketed features
extracted from the recipe, three clear a corrected bar once agent identity is
conditioned out, and they are worth 0.5-1.7 % of the residual variance each.
The claim one step up is the thesis: a model reading the *trajectory* knows
something those summaries do not. This measures that.

The task is pairwise, not absolute. Two runs from the same cell (same benchmark,
same base model) **and the same agent family**; which scored higher? Pairwise
removes calibration, puts the floor at exactly 0.5, and the within-family
restriction is the conditioning -- it asks what the trajectory adds once the
agent is known, which is the same question the 5-fold split asks.

Four things had to be controlled, and each is an arm rather than a caveat:

1. **The trajectory contains the answer.** 1004 of 1175 digests carry a
   score-shaped number next to an eval word -- the agent ran its own eval and
   printed it. A regex that just takes the largest such number reaches within-
   cell Spearman +0.51. So `selfreport` is an arm, and `redact` is the same
   digest with those numbers blanked. `raw` minus `redact` is how much of
   "reading the trajectory" is reading the answer off the page.
2. **Position.** An LLM ranking two long documents answers by position. Every
   pair is asked twice, A-first and B-first. The headline is accuracy on pairs
   answered *consistently*; the swap rate is reported next to it.
3. **The summaries themselves.** `features` fits the 21 bucketed features from
   `recipe_signal2.feat` on pair differences, leave-one-family-out, so the
   comparison is against a real fitted baseline rather than against chance.
4. **Ties.** The population is pairs whose accuracy gap is at least 0.05.
   Below that nobody can be right and including them caps every arm at once;
   the gap-stratified table is printed so the choice is visible.

Run:
    python3 tools/traj_read.py --arms selfreport,features         # free
    python3 tools/traj_read.py --arms recipe,redact,raw           # ~$270
"""

from __future__ import annotations

import argparse
import collections
import gzip
import itertools
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/splitdx")]

from awm import paths  # noqa: E402
from awm.analysis import recipe as RC  # noqa: E402

import battery as B  # noqa: E402
import recipe_signal2 as RS  # noqa: E402

RECIPES = ROOT / "splits/posttrainbench/recipes-tier1-v1.jsonl.gz"
OUTDIR = paths.data_root() / "trajread"
MODEL = "claude-opus-5"
PROJECT = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "sercan-v1")
REGION = os.environ.get("ANTHROPIC_VERTEX_REGION", "global")

#: both digests must fit one request with room to think, so both digest arms
#: use the same reduced budget -- half the extractor's 180k
DIGEST_BUDGET = 90_000
MIN_GAP = 0.05

# --- the self-report reader, used both as an arm and as the redactor ----------

_EVAL = (r"(?:accuracy|acc|score|eval|pass@?1|correct|exact.?match|solved"
         r"|em|f1|reward|win.?rate|bench)")
#: (pattern, is_percent). The scale has to come from the PATTERN, not from the
#: magnitude of what it matched. Dividing by 100 only when `v > 1` reads every
#: sub-1 percentage at fraction scale, and those exist: `pass@1 0.83%` became
#: 0.83, `win rate (stderr: +/-0.64%)` became 0.64, and a bare `1%` became a
#: PERFECT 1.000. Across the corpus that is 56 matches at the wrong scale, 20 of
#: the 1,175 run maxima, and 16 of the 42 runs that appeared to report a perfect
#: score. All 16 were inflated, never deflated, because the failure is one-sided.
SCORE_PAT_SCALED = [
    (re.compile(rf"({_EVAL}\D{{0,24}}?)(\d{{1,3}}(?:\.\d+)?)(\s*%)", re.I), True),
    # the `(?!\s*%)` matters: without it `pass@1 0.83%` matches BOTH patterns and
    # contributes 0.0083 and 0.83, and `max()` takes the wrong one.
    (re.compile(rf"({_EVAL}\D{{0,24}}?)(0?\.\d{{2,4}})()\b(?!\s*%)", re.I), False),
    (re.compile(rf"()(\d{{1,3}}(?:\.\d+)?)(\s*%\D{{0,24}}?{_EVAL})", re.I), True),
]
#: `redact` only blanks group 2 and never reads its value, so it is unaffected by
#: the scale and keeps using the bare patterns. The leak control is unchanged.
SCORE_PAT = [p for p, _ in SCORE_PAT_SCALED]


def self_scores(text: str) -> list[float]:
    """Every score-shaped number sitting next to a word meaning 'we evaluated'."""
    out = []
    for p, pct in SCORE_PAT_SCALED:
        for m in p.finditer(text):
            v = float(m.group(2))
            if pct:
                v /= 100.0
            if 0 <= v <= 1:
                out.append(v)
    return out


def redact(text: str) -> tuple[str, int]:
    """Blank those numbers, leave everything else -- including loss curves.

    Redaction is deliberately narrow and it is not complete: an agent that
    writes `61.3` on a line of its own, with no nearby eval word, survives it.
    That is why `selfreport` is re-run on the redacted text as a check, and why
    the honest reading of `redact` is a *lower* bound on leakage removed.
    """
    n = 0
    for p in SCORE_PAT:
        text, k = p.subn(lambda m: f"{m.group(1)}[REDACTED]{m.group(3)}", text)
        n += k
    return text, n


# --- population ---------------------------------------------------------------

def load_rows():
    rows = [json.loads(l) for l in gzip.open(RECIPES, "rt")]
    rows = [r for r in rows if isinstance(r.get("accuracy"), (int, float))]
    for r in rows:
        r["fam"] = B.agent_family(r["agent_model"])
        r.update(RS.feat(r))
    return rows


def build_pairs(rows, min_gap=MIN_GAP, seed=0):
    """Within cell, within agent family, gap at least `min_gap`.

    `a` and `b` are assigned by a seeded coin so the correct answer is not a
    function of catalogue order; the swap arm covers the rest.
    """
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["benchmark"], r["trained_model"], r["fam"])].append(r)
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(g):
        for x, y in itertools.combinations(sorted(g[k], key=lambda r: r["run"]), 2):
            if abs(x["accuracy"] - y["accuracy"]) < min_gap:
                continue
            a, b = (x, y) if rng.random() < 0.5 else (y, x)
            out.append({"id": f'{a["run"]}|{b["run"]}', "a": a, "b": b,
                        "cell": k[:2], "fam": k[2],
                        "gap": abs(a["accuracy"] - b["accuracy"]),
                        "truth": "A" if a["accuracy"] > b["accuracy"] else "B"})
    return out


# --- scoring ------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def report(name, pairs, picks):
    """picks: id -> (first_order_answer, swapped_order_answer) in {A,B,None}.

    Three accuracies, because an arm that abstains is not comparable to one
    that always answers: `acc-consistent` is over pairs answered the same way
    both ways round, `acc-answered` folds an inconsistent pair in as the coin
    flip it is, and `acc-pop` scores every pair in the population, giving an
    abstention 0.5. Only `acc-pop` compares arms with different coverage.
    """
    cons = wrong = incons = 0
    posA = 0
    n = 0
    for p in pairs:
        v = picks.get(p["id"])
        if not v:
            continue
        f, s = v
        if f is None or s is None:
            continue
        n += 1
        posA += (f == "A") + (s == "A")
        if f != s:
            incons += 1
        elif f == p["truth"]:
            cons += 1
        else:
            wrong += 1
    dec = cons + wrong
    N = len(pairs)
    acc = cons / dec if dec else float("nan")
    lo, hi = wilson(cons, dec)
    # an inconsistent answer and an abstention are both coin flips
    eff = (cons + 0.5 * incons) / n if n else float("nan")
    pop = (cons + 0.5 * (incons + N - n)) / N
    print(f'{name:>20} cover {n/N:5.1%}  self-consistent {dec/max(n,1):5.1%}  '
          f'acc-consistent {acc:6.1%} [{lo:.1%},{hi:.1%}]  '
          f'acc-answered {eff:6.1%}  acc-pop {pop:6.1%}  '
          f'says-A {posA/max(2*n,1):5.1%}')
    return {"arm": name, "n": n, "pop": N, "decided": dec, "correct": cons,
            "acc": acc, "ci": [lo, hi], "acc_answered": eff, "acc_pop": pop,
            "consistency": dec / n if n else float("nan"),
            "position_A": posA / (2 * n) if n else float("nan")}


def by_gap(name, pairs, picks):
    edges = [0.05, 0.10, 0.20, 0.40, 1.01]
    lo = 0.05
    print(f"      {'gap':>12} {'n':>5} {'acc':>7}")
    for e in edges[1:]:
        sub = [p for p in pairs if lo <= p["gap"] < e]
        c = d = 0
        for p in sub:
            v = picks.get(p["id"])
            if v and v[0] and v[1] and v[0] == v[1]:
                d += 1
                c += v[0] == p["truth"]
        if d:
            print(f"      {f'{lo:.2f}-{e:.2f}':>12} {d:5d} {c/d:7.1%}")
        lo = e


# --- arm: the self-report regex ------------------------------------------------

def _digest_of(row, budget=DIGEST_BUDGET):
    path = paths.events_dir("posttrainbench") / f'{row["experiment"]}__{row["run"]}.jsonl.gz'
    ev = RC.select(path, budget=budget)
    return RC.render(row["run"], ev, row, include_agent=False)


def _sr_worker(t):
    ex, run, do_redact = t
    path = paths.events_dir("posttrainbench") / f"{ex}__{run}.jsonl.gz"
    txt = RC.render(run, RC.select(path, budget=DIGEST_BUDGET), None,
                    include_agent=False)
    if do_redact:
        txt, _ = redact(txt)
    v = self_scores(txt)
    return run, (max(v) if v else None), len(v)


def arm_selfreport(pairs, rows, do_redact=False, workers=32):
    runs = {p[k]["run"]: p[k]["experiment"] for p in pairs for k in ("a", "b")}
    with ProcessPoolExecutor(workers) as ex:
        got = list(ex.map(_sr_worker,
                          [(v, k, do_redact) for k, v in runs.items()],
                          chunksize=8))
    best = {r: v for r, v, _ in got}
    found = {r: n for r, _, n in got}
    picks = {}
    for p in pairs:
        x, y = best.get(p["a"]["run"]), best.get(p["b"]["run"])
        if x is None or y is None or x == y:
            continue           # no number to read, or a tie -- an abstention
        w = "A" if x > y else "B"
        picks[p["id"]] = (w, w)   # deterministic, so order cannot matter
    cov = len(picks) / len(pairs)
    tot = sum(found.values())
    print(f"      {tot} score-shaped numbers over {len(runs)} runs; the rule "
          f"decides {cov:.1%} of pairs and abstains on the rest")
    return picks


# --- arms fitted on tabular features -------------------------------------------

#: how hard the agent worked, from the catalogue -- never from the trajectory.
#: This is the rival explanation for any trajectory-reading result: a model
#: shown a run that crashed at turn 4 against one that ran 300 turns does not
#: need to understand the recipe to pick the second. Bucketed like everything
#: else, because a raw count is a near-unique categorical.
EFFORT_SRC = ("num_turns", "duration_ms", "total_cost_usd", "time_taken")
#: written under a prefix, never over the source column -- bucketing in place
#: makes the function non-idempotent, and the second call then reads its own
#: string output as "not a number" and quietly returns an all-constant block
EFFORT = tuple("work_" + k for k in EFFORT_SRC)


def effort_feat(r):
    out = {}
    for k in EFFORT_SRC:
        v = r.get(k)
        out["work_" + k] = (RS._log_bucket(v)
                            if isinstance(v, (int, float)) else "na")
    return out


def arm_features(pairs, rows, which="recipe"):
    """Leave-one-family-out logistic regression on one-hot feature differences.

    Held out by family, because the whole question is what survives when the
    agent is unknown; a random split would let the model memorise a family.
    ``which`` selects the feature block: the 21 recipe summaries, the 4 effort
    columns, or both.
    """
    import warnings

    from scipy.optimize import OptimizeWarning
    from sklearn.linear_model import LogisticRegression

    warnings.simplefilter("ignore", OptimizeWarning)
    for r in rows:
        r.update(effort_feat(r))
    blocks = {"recipe": [f for f in RS.feat(rows[0]) if f not in RS.SKIP3],
              "effort": list(EFFORT)}
    blocks["both"] = blocks["recipe"] + blocks["effort"]
    feats = blocks[which]
    levels = {f: sorted({r[f] for r in rows}) for f in feats}
    index = {f: {v: i for i, v in enumerate(levels[f])} for f in feats}
    width = sum(len(levels[f]) for f in feats)
    offs, o = {}, 0
    for f in feats:
        offs[f] = o
        o += len(levels[f])

    def vec(r):
        v = np.zeros(width)
        for f in feats:
            v[offs[f] + index[f][r[f]]] = 1.0
        return v

    cache = {r["run"]: vec(r) for r in rows}
    X = np.array([cache[p["a"]["run"]] - cache[p["b"]["run"]] for p in pairs])
    y = np.array([p["truth"] == "A" for p in pairs], dtype=int)
    fam = np.array([p["fam"] for p in pairs])

    picks = {}
    for held in sorted(set(fam)):
        tr, te = fam != held, fam == held
        if tr.sum() < 30 or te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        # antisymmetric problem: (x, y) and (-x, 1-y) are the same fact, so
        # train on both and drop the intercept -- otherwise the model learns
        # the base rate of "A wins", which is 50% by construction and useless
        Xa = np.vstack([X[tr], -X[tr]])
        ya = np.concatenate([y[tr], 1 - y[tr]])
        m = LogisticRegression(max_iter=4000, C=0.1, fit_intercept=False)
        m.fit(Xa, ya)
        pr = m.predict_proba(X[te])[:, 1]
        for p, q in zip([p for p, t in zip(pairs, te) if t], pr):
            w = "A" if q > 0.5 else "B"
            picks[p["id"]] = (w, w)
    print(f"      {len(feats)} features, {width} one-hot columns, "
          f"{len(set(fam))} held-out families, {len(picks)} pairs predicted")
    return picks


# --- arms that call the model ---------------------------------------------------

TOOL = {
    "name": "pick",
    "description": "Say which run scored higher.",
    "input_schema": {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B"]},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            "why": {"type": "string",
                    "description": "One sentence, naming the concrete "
                                   "difference you decided on."},
        },
        "required": ["winner", "confidence", "why"],
    },
}

SYSTEM = """\
You are shown two attempts at the SAME post-training task -- the same benchmark, \
the same base model, and the same agent model wrote both. Exactly one of them \
scored higher on the held-out benchmark. Say which.

Because the task, the base model and the agent are held fixed, generic quality \
signals will not separate these two. What is left is the recipe: which data was \
used and in what proportion, how much of it, how many stages, what the \
optimiser settings were, what was tried and abandoned, and whether the training \
distribution matches what the benchmark actually asks for.

You are not being shown the benchmark result and you must not pretend to \
remember it. Decide from the recipe. If a number that looks like a benchmark \
score appears in the text it is the agent's own local measurement -- it may be \
on a different sample, a different prompt, or a checkpoint that was not the one \
shipped.

Guessing is required: there is no "cannot tell" option. Use `confidence` to say \
how thin the evidence is, and `why` to name the one concrete difference you \
decided on, so a disagreement can be traced.
"""


class Judge:
    def __init__(self, tries=6):
        from anthropic import AnthropicVertex
        self.c = AnthropicVertex(project_id=PROJECT, region=REGION)
        self.tries = tries
        self.usage = collections.Counter()
        self.lock = threading.Lock()

    def ask(self, first: str, second: str) -> tuple[str | None, dict]:
        msg = (f"=== RUN A ===\n{first}\n\n=== RUN B ===\n{second}\n\n"
               f"Which run, A or B, scored higher on the held-out benchmark?")
        for n in range(self.tries):
            try:
                with self.c.messages.stream(
                    model=MODEL, max_tokens=16_000,
                    system=[{"type": "text", "text": SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": msg}],
                    tools=[TOOL], tool_choice={"type": "tool", "name": "pick"},
                    thinking={"type": "adaptive"},
                ) as s:
                    m = s.get_final_message()
                with self.lock:
                    for k in ("input_tokens", "output_tokens",
                              "cache_creation_input_tokens",
                              "cache_read_input_tokens"):
                        self.usage[k] += getattr(m.usage, k, 0) or 0
                    self.usage["calls"] += 1
                tu = next((b for b in m.content if b.type == "tool_use"), None)
                if tu is None:
                    return None, {"stop": m.stop_reason}
                return tu.input.get("winner"), tu.input
            except Exception as e:  # noqa: BLE001
                if "invalid_request" in str(e) or n == self.tries - 1:
                    return None, {"error": f"{type(e).__name__}: {e}"[:300]}
                time.sleep(min(60, 2 ** n))
        return None, {}


def _text_worker(t):
    kind, ex, run, meta = t
    if kind == "summary":
        # the same 21 bucketed values the fitted arm sees, as text. Its only
        # purpose is to separate two explanations for `features` scoring at
        # chance: that bucketing destroyed the information, or that 540 pairs
        # is not enough to learn it. A model needs no training rows.
        return run, "\n".join(f"{k}: {v}" for k, v in meta.items())
    if kind == "recipe":
        txt = json.dumps(meta, ensure_ascii=False, indent=1)
        txt, _ = redact(txt)
        return run, txt
    path = paths.events_dir("posttrainbench") / f"{ex}__{run}.jsonl.gz"
    txt = RC.render(run, RC.select(path, budget=DIGEST_BUDGET), meta,
                    include_agent=False)
    if kind == "redact":
        txt, _ = redact(txt)
    return run, txt


#: fields the extraction record must lose before it is shown -- everything the
#: catalogue knows, plus the outcome itself
DROP = {"run", "experiment", "benchmark", "trained_model", "agent_model",
        "trace_format", "seed", "time_budget_h", "time_taken", "accuracy",
        "stderr", "total_cost_usd", "num_turns", "duration_ms", "extraction",
        "fam"}


def arm_model(kind, pairs, rows, out_path, workers=24):
    byrun = {r["run"]: r for r in rows}
    runs = {}
    featkeys = set(RS.feat(rows[0]))
    for p in pairs:
        for k in ("a", "b"):
            r = p[k]
            if kind == "recipe":
                meta = {k2: v for k2, v in r.items()
                        if k2 not in DROP and k2 not in featkeys}
            elif kind == "summary":
                meta = {k2: r[k2] for k2 in sorted(featkeys)
                        if k2 not in RS.SKIP3}
            else:
                meta = {k2: r.get(k2) for k2 in RC.HEADER_KEYS}
            runs[r["run"]] = (kind, r["experiment"], r["run"], meta)
    with ProcessPoolExecutor(16) as ex:
        text = dict(ex.map(_text_worker, list(runs.values()), chunksize=4))
    ch = [len(v) for v in text.values()]
    print(f"      {len(text)} texts, median {int(np.median(ch)):,} chars, "
          f"max {max(ch):,}")

    done = {}
    if out_path.exists():
        for line in out_path.open():
            try:
                d = json.loads(line)
                done[(d["id"], d["order"])] = d
            except Exception:  # noqa: BLE001
                pass
    jobs = [(p, o) for p in pairs for o in (0, 1)
            if (p["id"], o) not in done]
    print(f"      {len(done)} calls cached, {len(jobs)} to make")

    j = Judge()
    fh = out_path.open("a")
    wl, st = threading.Lock(), collections.Counter()
    t0 = time.time()

    def one(job):
        p, order = job
        x, y = (p["a"], p["b"]) if order == 0 else (p["b"], p["a"])
        w, raw = j.ask(text[x["run"]], text[y["run"]])
        # `w` names the slot; translate back to the pair's own A/B
        if w in ("A", "B") and order == 1:
            w = "B" if w == "A" else "A"
        rec = {"id": p["id"], "order": order, "winner": w,
               "confidence": raw.get("confidence"), "why": raw.get("why"),
               "truth": p["truth"], "gap": p["gap"], "fam": p["fam"]}
        with wl:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            st["n"] += 1
            st["ok"] += w is not None
            if st["n"] % 50 == 0:
                el = time.time() - t0
                print(f'      {st["n"]}/{len(jobs)} {el/60:.1f}min '
                      f'ok={st["ok"]} in={j.usage["input_tokens"]/1e6:.1f}M',
                      flush=True)
        return rec

    if jobs:
        with ThreadPoolExecutor(workers) as tp:
            list(tp.map(one, jobs))
    fh.close()
    for line in out_path.open():
        d = json.loads(line)
        done[(d["id"], d["order"])] = d
    picks = {}
    for p in pairs:
        f = done.get((p["id"], 0), {}).get("winner")
        s = done.get((p["id"], 1), {}).get("winner")
        picks[p["id"]] = (f, s)
    u = j.usage
    if u["calls"]:
        cost = (u["input_tokens"] * 5 + u["cache_creation_input_tokens"] * 6.25
                + u["cache_read_input_tokens"] * 0.5
                + u["output_tokens"] * 25) / 1e6
        print(f'      {u["calls"]} calls, ~${cost:,.2f}')
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="selfreport,features")
    ap.add_argument("--min-gap", type=float, default=MIN_GAP)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    rows = load_rows()
    pairs = build_pairs(rows, args.min_gap)
    if args.limit:
        step = max(1, len(pairs) // args.limit)
        pairs = pairs[::step][: args.limit]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} runs -> {len(pairs)} pairs "
          f"(same cell, same agent family, accuracy gap >= {args.min_gap}); "
          f'{len({p["fam"] for p in pairs})} families, '
          f'{len({p["cell"] for p in pairs})} cells')
    print(f'  gap median {np.median([p["gap"] for p in pairs]):.3f}, '
          f'truth is A in {sum(p["truth"] == "A" for p in pairs)/len(pairs):.1%} '
          f"of pairs\n")

    print(f'{"arm":>12} {"n":>5}')
    res, allpicks = [], {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        if arm == "selfreport":
            picks = arm_selfreport(pairs, rows)
        elif arm == "selfreport-redacted":
            picks = arm_selfreport(pairs, rows, do_redact=True)
        elif arm in ("features", "effort", "features+effort"):
            picks = arm_features(pairs, rows,
                                 {"features": "recipe", "effort": "effort",
                                  "features+effort": "both"}[arm])
        elif arm in ("summary", "recipe", "redact", "raw"):
            picks = arm_model(arm, pairs, rows, OUTDIR / f"{arm}.jsonl",
                              args.workers)
        else:
            print(f"  unknown arm {arm}")
            continue
        allpicks[arm] = picks
        res.append(report(arm, pairs, picks))
        by_gap(arm, pairs, picks)
        print()

    (OUTDIR / "summary.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
