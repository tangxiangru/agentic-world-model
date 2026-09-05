#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K from OpenMathInstruct-2 (gsm8k-derived rows)
plus the original GSM8K *train* split.

Output: jsonl with {id, question, solution, nshot, source}
`solution` is the assistant target text WITHOUT the stop token; the trainer
appends <end_of_turn>. Every solution ends with a single "ANSWER: <int>" line,
which is also the last number in the text (the grader takes the last number).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
GSM8K_GLOB = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

INT_RE = re.compile(r"^-?\d+$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CALC_RE = re.compile(r"<<[^>]*>>")
STOP_TOKEN = "<end_of_turn>"


def norm_int(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    if s.endswith(".0"):
        s = s[:-2]
    if INT_RE.match(s):
        return str(int(s))
    return None


def clean_omi2(sol: str, ans: str) -> str | None:
    """Drop the \\boxed wrapper (keep the value) and append the ANSWER line."""
    if "\\boxed" not in sol:
        return None
    sol = BOXED_RE.sub(r"\1", sol)
    if "\\boxed" in sol:  # nested / malformed
        return None
    sol = sol.strip()
    return f"{sol}\n\nANSWER: {ans}"


def clean_gsm8k(sol: str, ans: str) -> str:
    body = sol.split("####")[0]
    body = CALC_RE.sub("", body).strip()
    return f"{body}\n\nANSWER: {ans}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--omi2-glob", default=OMI2_GLOB)
    ap.add_argument("--max-aug", type=int, default=90000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-direct", type=int, default=10**9)
    ap.add_argument("--max-per-problem-direct", type=int, default=None)
    ap.add_argument("--nshot-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats: Counter[str] = Counter()
    out_rows: list[dict] = []
    seen_sol: set[str] = set()
    per_problem: dict[str, int] = defaultdict(int)

    # ---- OpenMathInstruct-2, gsm8k-derived rows only -----------------------
    files = sorted(glob.glob(args.omi2_glob))
    assert files, args.omi2_glob
    direct, aug = [], []
    for f in files:
        tbl = pq.read_table(f)
        for r in tbl.to_pylist():
            src = r["problem_source"]
            if "gsm8k" not in src:
                stats["skip_not_gsm8k"] += 1
                continue
            ans = norm_int(r["expected_answer"] or "")
            if ans is None:
                stats["skip_non_int_answer"] += 1
                continue
            sol = clean_omi2(r["generated_solution"] or "", ans)
            if sol is None:
                stats["skip_no_boxed"] += 1
                continue
            rec = {"question": r["problem"].strip(), "solution": sol, "source": src}
            (direct if src == "gsm8k" else aug).append(rec)
        del tbl

    rng.shuffle(direct)
    rng.shuffle(aug)
    stats["omi2_direct_avail"] = len(direct)
    stats["omi2_aug_avail"] = len(aug)

    def take(rows, budget, cap=None):
        cap = args.max_per_problem if cap is None else cap
        n = 0
        for rec in rows:
            if n >= budget:
                break
            key = rec["solution"]
            if key in seen_sol:
                stats["skip_dup_solution"] += 1
                continue
            if per_problem[rec["question"]] >= cap:
                stats["skip_per_problem_cap"] += 1
                continue
            seen_sol.add(key)
            per_problem[rec["question"]] += 1
            out_rows.append(rec)
            n += 1
        return n

    stats["taken_omi2_direct"] = take(direct, args.max_direct, args.max_per_problem_direct)
    stats["taken_omi2_aug"] = take(aug, args.max_aug)

    # ---- original GSM8K train split ---------------------------------------
    gfiles = sorted(glob.glob(GSM8K_GLOB))
    assert gfiles, GSM8K_GLOB
    n_g = 0
    for f in gfiles:
        for r in pq.read_table(f).to_pylist():
            ans = norm_int(r["answer"].split("####")[-1])
            if ans is None:
                stats["skip_gsm8k_non_int"] += 1
                continue
            out_rows.append(
                {
                    "question": r["question"].strip(),
                    "solution": clean_gsm8k(r["answer"], ans),
                    "source": "gsm8k_train_original",
                }
            )
            n_g += 1
    stats["taken_gsm8k_original"] = n_g

    rng.shuffle(out_rows)
    n_shot_rows = int(len(out_rows) * args.nshot_frac)
    for i, rec in enumerate(out_rows):
        rec["nshot"] = rng.randint(2, 10) if i < n_shot_rows else 0
    rng.shuffle(out_rows)  # so that a --limit prefix is a representative sample
    for i, rec in enumerate(out_rows):
        rec["id"] = f"sft-{i:07d}"

    # `completion` is the literal training target, terminator included, so the
    # protocol's stop_token / answer_marker / max_seq_len checks can read it.
    with open(args.out, "w") as fh:
        for rec in out_rows:
            rec["completion"] = rec.pop("solution").strip() + STOP_TOKEN
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats["TOTAL"] = len(out_rows)
    for k, v in sorted(stats.items()):
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
