"""Build the SFT corpus for GSM8K from nvidia/OpenMathInstruct-2.

Target shape is the one the grader reads:
    <reasoning steps>\n\nANSWER: <number>
and the turn is terminated by <end_of_turn>, the terminator in templates/gemma3.jinja.

Every row is verified with inspect_ai's own scorer before it is written.
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import pandas as pd
import pyarrow.parquet as pq
from inspect_ai.scorer._common import match_str

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
LATEX_HINT_RE = re.compile(r"\\(frac|sqrt|begin|sum|int|pi|cdot|times|le|ge|neq|in|mathbb|text)")


def clean_solution(sol: str, answer: str) -> str | None:
    """Strip LaTeX \\boxed{} wrappers and normalise whitespace."""
    if "\\boxed" not in sol:
        return None
    # unwrap every \boxed{...}
    prev = None
    while prev != sol:
        prev = sol
        sol = BOXED_RE.sub(lambda m: m.group(1), sol)
    if "\\boxed" in sol:  # nested braces we could not unwrap
        return None
    sol = sol.replace("\\$", "$").replace("\\%", "%")
    sol = re.sub(r"[ \t]+\n", "\n", sol).strip()
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol or None


def normalise_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    if not NUM_RE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    # reject silly magnitudes / degenerate answers
    try:
        v = float(a)
    except ValueError:
        return None
    if abs(v) > 1e12:
        return None
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_all.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--math-frac", type=float, default=0.12,
                    help="share of the final corpus drawn from MATH-derived problems")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    print(f"{len(files)} shards")

    gsm_rows: list[dict] = []
    math_rows: list[dict] = []
    per_problem: dict[str, int] = defaultdict(int)

    stats = defaultdict(int)
    for f in files:
        df = pq.read_table(f).to_pandas()
        for src, prob, sol, ans in zip(df["problem_source"], df["problem"],
                                       df["generated_solution"], df["expected_answer"]):
            stats["seen"] += 1
            is_gsm = src in ("gsm8k", "augmented_gsm8k")
            if not is_gsm and rng.random() > 0.25:
                continue  # subsample MATH-derived rows early, they are the bulk
            a = normalise_answer(ans)
            if a is None:
                stats["bad_answer"] += 1
                continue
            body = clean_solution(sol, a)
            if body is None:
                stats["no_boxed"] += 1
                continue
            if is_gsm and LATEX_HINT_RE.search(body):
                stats["latex_in_gsm"] += 1
                continue
            if len(body) < 40 or len(body) > 3500:
                stats["length"] += 1
                continue
            key = prob.strip()
            if per_problem[key] >= args.max_per_problem:
                stats["cap"] += 1
                continue
            target = f"{body}\n\nANSWER: {a}"
            # the grader takes the LAST numeric token of the completion
            _, ok = match_str(value=target, target=a, location="end", numeric=True)
            if not ok:
                stats["scorer_reject"] += 1
                continue
            if target.count("ANSWER:") != 1:
                stats["multi_marker"] += 1
                continue
            per_problem[key] += 1
            rec = {"question": key, "target": target, "answer": a, "source": src}
            (gsm_rows if is_gsm else math_rows).append(rec)
        print(f"  {os.path.basename(f)}: gsm={len(gsm_rows)} math={len(math_rows)}", flush=True)

    rng.shuffle(gsm_rows)
    rng.shuffle(math_rows)
    n_math = int(len(gsm_rows) * args.math_frac / max(1e-9, 1 - args.math_frac))
    math_rows = math_rows[:n_math]
    rows = gsm_rows + math_rows
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(dict(stats), indent=2))
    print(f"wrote {len(rows)} rows ({len(gsm_rows)} gsm-derived, {len(math_rows)} math-derived) -> {args.out}")


if __name__ == "__main__":
    main()
