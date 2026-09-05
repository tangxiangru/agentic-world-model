#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Sources (all derived from *training* splits only):
  * nvidia/OpenMathInstruct-2, rows whose problem_source is gsm8k or
    augmented_gsm8k (seeded from the GSM8K train split).
  * openai/gsm8k, main/train split, original human solutions.

Every target is normalised to: <reasoning> \n ANSWER: <number>
so the grader (inspect match(numeric=True, location="end")) reads exactly one
number at the end of the completion.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

OMI_DIR = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
    "snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data"
)
GSM8K_TRAIN = (
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/"
    "snapshots/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"
)

NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
CALC_RE = re.compile(r"<<[^>]*>>")


def clean_number(s: str) -> str | None:
    s = s.strip().replace(",", "").rstrip(".")
    if not NUM_RE.match(s):
        return None
    if s.startswith("-"):
        return None
    # keep integers and simple decimals; drop absurdly long ones
    if len(s) > 12:
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or None


def normalise_omi(sol: str, ans: str) -> str | None:
    """Strip \\boxed{}, drop latex noise, append the ANSWER line."""
    if "\\boxed" not in sol:
        return None
    body = BOXED_RE.sub(r"\1", sol)
    body = body.replace("\\[", "").replace("\\]", "")
    body = body.replace("\\(", "").replace("\\)", "")
    body = body.replace("$", "")
    body = body.replace("\\%", "%").replace("\\$", "$")
    body = re.sub(r"[ \t]+\n", "\n", body).strip()
    if not body:
        return None
    # any remaining latex command is a sign this is a MATH-style row
    if "\\" in body:
        return None
    if "ANSWER" in body.upper():
        return None
    return f"{body}\nANSWER: {ans}"


def normalise_gsm8k(sol: str, ans: str) -> str | None:
    body = CALC_RE.sub("", sol)
    body = body.split("####")[0].strip()
    if not body or "ANSWER" in body.upper():
        return None
    return f"{body}\nANSWER: {ans}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-problems", type=int, default=70000)
    ap.add_argument("--gsm8k-train-repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/train_main.jsonl")
    ap.add_argument("--stats-out", default="data/train_main.stats.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_problem: dict[str, list[str]] = defaultdict(list)
    seen_sol: set[int] = set()
    stats = {"omi_rows_read": 0, "omi_gsm8k_rows": 0, "omi_kept": 0, "gsm8k_kept": 0}

    for i in range(args.shards):
        path = f"{OMI_DIR}/train-{i:05d}-of-00032.parquet"
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["problem", "generated_solution",
                                              "expected_answer", "problem_source"]):
            src = batch.column("problem_source").to_pylist()
            prob = batch.column("problem").to_pylist()
            sol = batch.column("generated_solution").to_pylist()
            ans = batch.column("expected_answer").to_pylist()
            stats["omi_rows_read"] += len(src)
            for s, p, so, a in zip(src, prob, sol, ans):
                if s not in ("gsm8k", "augmented_gsm8k"):
                    continue
                stats["omi_gsm8k_rows"] += 1
                p = p.strip()
                if len(by_problem[p]) >= args.max_per_problem:
                    continue
                if not by_problem[p] and len(by_problem) > args.max_problems:
                    del by_problem[p]
                    continue
                n = clean_number(a)
                if n is None:
                    continue
                t = normalise_omi(so, n)
                if t is None or len(t) > 4000:
                    continue
                h = hash((p, t))
                if h in seen_sol:
                    continue
                seen_sol.add(h)
                by_problem[p].append(t)
                stats["omi_kept"] += 1
        print(f"shard {i}: problems={len(by_problem)} kept={stats['omi_kept']}", flush=True)

    rows = [{"question": q, "answer": t, "source": "openmathinstruct2_gsm8k"}
            for q, ts in by_problem.items() for t in ts]

    # original GSM8K train solutions
    gt = pq.read_table(GSM8K_TRAIN).to_pylist()
    for _ in range(args.gsm8k_train_repeats):
        for r in gt:
            n = clean_number(r["answer"].split("####")[-1])
            if n is None:
                continue
            t = normalise_gsm8k(r["answer"], n)
            if t is None:
                continue
            rows.append({"question": r["question"].strip(), "answer": t,
                         "source": "gsm8k_train_human"})
            stats["gsm8k_kept"] += 1

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats["n_rows"] = len(rows)
    stats["n_problems"] = len(by_problem)
    with open(args.stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
