#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K from OpenMathInstruct-2 (gsm8k-derived rows).

Output: data/sft_<tag>.jsonl with fields
    {"question", "target", "n_shot", "source"}
plus data/sft_<tag>.decon.jsonl with {"question","answer"} for the
contamination checker.

No GSM8K *test* item is read anywhere in this file; the only gsm8k data used
is the TRAIN split (directly, and as the seed of OpenMathInstruct-2).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

from common_fmt import STOP_TOKEN

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-*.parquet"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
CALC = re.compile(r"<<[^>]*>>")
INT_RE = re.compile(r"^-?\d+$")


def clean_solution(sol: str) -> str | None:
    """Strip \\boxed{} wrappers and calculator annotations; reject leftovers."""
    prev = None
    while prev != sol:
        prev = sol
        sol = BOXED.sub(r"\1", sol)
    if "\\boxed" in sol or "####" in sol:
        return None
    sol = CALC.sub("", sol)
    sol = sol.replace("$\\", "").strip()
    return sol


def load_gsm8k_train():
    path = sorted(glob.glob(GSM8K_TRAIN))[0]
    t = pq.read_table(path).to_pylist()
    out = []
    for r in t:
        q = r["question"].strip()
        body, _, ans = r["answer"].rpartition("####")
        out.append((q, CALC.sub("", body).strip(), ans.strip().replace(",", "")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm8k-src", type=int, default=15000)
    ap.add_argument("--n-aug-src", type=int, default=11000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-per-problem-aug", type=int, default=None)
    ap.add_argument("--fewshot-frac-4", type=float, default=0.10)
    ap.add_argument("--fewshot-frac-10", type=float, default=0.10)
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm_train = load_gsm8k_train()
    print(f"gsm8k train exemplar pool: {len(gsm_train)}")

    by_problem: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    kept_src = {"gsm8k": 0, "augmented_gsm8k": 0}
    files = sorted(glob.glob(OMI_GLOB))
    print(f"{len(files)} OpenMathInstruct-2 shards")
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        for r in t.to_pylist():
            src = r["problem_source"]
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            ans = (r["expected_answer"] or "").strip().replace(",", "")
            if not INT_RE.match(ans):
                continue
            sol = clean_solution(r["generated_solution"] or "")
            if not sol or len(sol) > args.max_sol_chars or len(sol) < 40:
                continue
            if ans not in sol:
                continue
            q = (r["problem"] or "").strip()
            if not q:
                continue
            cap = args.max_per_problem if src == 'gsm8k' else (
                args.max_per_problem_aug if args.max_per_problem_aug is not None else args.max_per_problem)
            if len(by_problem[q]) >= cap:
                continue
            by_problem[q].append((sol, ans, src))
            kept_src[src] += 1
    print("candidate solutions by source:", kept_src, "unique problems:", len(by_problem))

    pools = {"gsm8k": [], "augmented_gsm8k": []}
    for q, sols in by_problem.items():
        for sol, ans, src in sols:
            pools[src].append((q, sol, ans, src))
    for k in pools:
        rng.shuffle(pools[k])
    rows = pools["gsm8k"][: args.n_gsm8k_src] + pools["augmented_gsm8k"][: args.n_aug_src]
    rng.shuffle(rows)
    print(f"selected {len(rows)} rows "
          f"({min(len(pools['gsm8k']), args.n_gsm8k_src)} gsm8k / "
          f"{min(len(pools['augmented_gsm8k']), args.n_aug_src)} augmented)")

    n4 = int(len(rows) * args.fewshot_frac_4)
    n10 = int(len(rows) * args.fewshot_frac_10)
    with open(args.out, "w") as fo, open(args.out.replace(".jsonl", ".decon.jsonl"), "w") as fd:
        for i, (q, sol, ans, src) in enumerate(rows):
            n_shot = 4 if i < n4 else (10 if i < n4 + n10 else 0)
            shots = []
            if n_shot:
                shots = [gsm_train[j] for j in rng.sample(range(len(gsm_train)), n_shot)]
            target = f"{sol}\n\nANSWER: {ans}" + STOP_TOKEN
            fo.write(json.dumps({
                "question": q, "target": target, "n_shot": n_shot,
                "shots": shots, "source": src, "answer": ans,
            }) + "\n")
            fd.write(json.dumps({"question": q, "answer": target}) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows; {n4} 4-shot, {n10} 10-shot)")


if __name__ == "__main__":
    main()
