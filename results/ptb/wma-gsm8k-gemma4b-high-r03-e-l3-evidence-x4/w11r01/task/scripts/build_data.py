#!/usr/bin/env python3
"""Build GSM8K-style SFT data from nvidia/OpenMathInstruct-2 (gsm8k-derived rows only).

Every target is reshaped for this harness: the chain of thought is kept, the
LaTeX \\boxed{} marker is unwrapped, and the target closes with a single
"ANSWER: <n>" line -- the only answer marker, and the last numeric token, which
is what inspect's match(numeric=True) scorer reads.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
WS = re.compile(r"\s+")
# the terminator templates/gemma3.jinja closes an assistant turn with (token id 106)
END_OF_TURN = "<end_of_turn>"


def norm_answer(a: str) -> str | None:
    """Return a clean plain-number string, or None if the answer is not a number."""
    a = a.strip().replace("$", "").replace(",", "").replace("%", "").strip()
    a = a.rstrip(".")
    if not NUMERIC.match(a):
        return None
    if "." in a:
        f = float(a)
        if f.is_integer():
            return str(int(f))
        return ("%f" % f).rstrip("0").rstrip(".")
    return str(int(a))


def clean_solution(sol: str) -> str | None:
    if sol.count("\\boxed") != 1:
        return None
    sol = BOXED.sub(lambda m: m.group(1), sol)
    if "\\boxed" in sol or "\\[" in sol or "\\frac" in sol or "\\text" in sol:
        return None
    sol = sol.replace("$", "").strip()
    # a stray "####" would be a second answer marker the grader could pick up
    if "####" in sol or "ANSWER:" in sol:
        return None
    return sol


def norm_problem(p: str) -> str:
    return WS.sub(" ", p.strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--per-problem", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--min-sol-chars", type=int, default=80)
    ap.add_argument("--split", default="train_1M")
    ap.add_argument("--exclude", default=None, help="jsonl whose questions must not appear in the output")
    args = ap.parse_args()

    files = sorted(glob.glob(
        f"/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/{args.split}-*.parquet"
    ))
    assert files, "OpenMathInstruct-2 train_1M parquets not found"

    import pyarrow.parquet as pq

    by_problem: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    n_seen = n_kept = 0
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        for prob, sol, ans, src in zip(cols["problem"], cols["generated_solution"],
                                       cols["expected_answer"], cols["problem_source"]):
            if "gsm8k" not in src:
                continue
            n_seen += 1
            a = norm_answer(ans)
            if a is None:
                continue
            s = clean_solution(sol)
            if s is None or not (args.min_sol_chars <= len(s) <= args.max_sol_chars):
                continue
            # the unwrapped boxed value must still be the answer we will assert
            if norm_answer(a) is None:
                continue
            by_problem[norm_problem(prob)].append((prob, s, a))
            n_kept += 1

    exclude = set()
    if args.exclude:
        for line in open(args.exclude):
            exclude.add(norm_problem(json.loads(line)["question"]))

    rng = random.Random(args.seed)
    rows = []
    for key, cands in by_problem.items():
        if key in exclude:
            continue
        rng.shuffle(cands)
        for prob, s, a in cands[: args.per_problem]:
            rows.append({
                "question": prob,
                "target": s + "\n\nANSWER: " + a + END_OF_TURN,
                "answer": a,
            })
    rng.shuffle(rows)
    rows = rows[: args.n]

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # a checker-shaped copy: one document per line, question + solution together
    with open(args.out.replace(".jsonl", "") + ".decon.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")

    print(f"split={args.split} excluded_problems={len(exclude)} gsm8k-derived rows seen={n_seen} kept={n_kept} "
          f"unique_problems={len(by_problem)} written={len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
