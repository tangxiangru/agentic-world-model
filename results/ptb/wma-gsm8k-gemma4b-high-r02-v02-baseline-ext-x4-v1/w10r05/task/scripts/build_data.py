"""Build the SFT corpus for GSM8K.

Sources (both GSM8K-TRAIN-derived only; the gsm8k TEST split is never touched):
  * openai/gsm8k main @740312a  -- train split, human reference chains
  * nvidia/OpenMathInstruct-2 @469216e -- train_1M shards, rows whose
    problem_source is 'gsm8k' (original train problems, 405B-written solutions)
    or 'augmented_gsm8k' (new problems seeded from train problems)

Output rows are {"prompt", "completion"} where prompt is the exact string the
grader's chat template produces and completion ends with

    ANSWER: <n><end_of_turn>

A held-out slice of the gsm8k TRAIN split is written to dev_train.jsonl and is
excluded from training, so there is a dev signal that never touches the test set.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import EOT, fewshot_system, render_prompt  # noqa: E402

OMI2_DIR = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data/"
)
OMI2_SHARDS = [f"train_1M-0000{i}-of-00003.parquet" for i in range(3)]

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
CALC = re.compile(r"<<[^>]*>>")
INT_RE = re.compile(r"^-?\d+(\.\d+)?$")


def clean_solution(sol: str) -> str:
    sol = BOXED.sub(r"\1", sol)
    sol = sol.replace("\\[", "").replace("\\]", "")
    return sol.strip()


def norm_problem(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/ben/task/data")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-augmented", type=int, default=40000)
    ap.add_argument("--fewshot-frac", type=float, default=0.06)
    ap.add_argument("--holdout", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- gsm8k train: held-out dev slice + few-shot demo pool ---------------
    gsm = load_dataset("openai/gsm8k", "main")["train"]
    idx = list(range(len(gsm)))
    rng.shuffle(idx)
    holdout_idx = set(idx[: args.holdout])
    holdout_problems = {norm_problem(gsm[i]["question"]) for i in holdout_idx}

    dev_path = os.path.join(args.out_dir, "dev_train.jsonl")
    with open(dev_path, "w") as f:
        for i in sorted(holdout_idx):
            q = gsm[i]["question"]
            a = gsm[i]["answer"].split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"id": f"train-{i}", "question": q, "gold": a}) + "\n")
    print(f"dev holdout -> {dev_path} ({len(holdout_idx)})")

    demo_pool = []
    for i in idx[args.holdout : args.holdout + 400]:
        ans = gsm[i]["answer"]
        reasoning = CALC.sub("", ans.split("####")[0]).strip()
        target = ans.split("####")[-1].strip().replace(",", "")
        demo_pool.append((gsm[i]["question"], reasoning, target))

    # ---- OpenMathInstruct-2, gsm8k-derived rows ----------------------------
    per_problem: dict[str, int] = collections.Counter()
    orig_rows, aug_rows = [], []
    skipped = collections.Counter()
    for shard in OMI2_SHARDS:
        pf = pq.ParquetFile(OMI2_DIR + shard)
        for batch in pf.iter_batches(batch_size=50000):
            for r in batch.to_pylist():
                src = r["problem_source"]
                if src not in ("gsm8k", "augmented_gsm8k"):
                    continue
                ans = (r["expected_answer"] or "").strip().replace(",", "")
                if not INT_RE.match(ans):
                    skipped["non_numeric_answer"] += 1
                    continue
                key = norm_problem(r["problem"])
                if key in holdout_problems:
                    skipped["holdout"] += 1
                    continue
                if per_problem[key] >= args.max_per_problem:
                    skipped["per_problem_cap"] += 1
                    continue
                sol = clean_solution(r["generated_solution"])
                if "ANSWER:" in sol or "answer:" in sol.lower():
                    skipped["marker_in_body"] += 1
                    continue
                if len(sol) < 20 or len(sol) > 4000:
                    skipped["length"] += 1
                    continue
                per_problem[key] += 1
                row = {"problem": r["problem"].strip(), "solution": sol, "answer": ans}
                (orig_rows if src == "gsm8k" else aug_rows).append(row)
    print(f"omi2: gsm8k={len(orig_rows)} augmented={len(aug_rows)} skipped={dict(skipped)}")

    rng.shuffle(aug_rows)
    aug_rows = aug_rows[: args.n_augmented]
    rows = orig_rows + aug_rows
    rng.shuffle(rows)

    # ---- render ------------------------------------------------------------
    n_fewshot = 0
    out_path = os.path.join(args.out_dir, "sft_train.jsonl")
    k_choices = [2, 2, 3, 4, 4, 6, 8, 10]
    with open(out_path, "w") as f:
        for row in rows:
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.choice(k_choices)
                system = fewshot_system(rng.sample(demo_pool, k))
                n_fewshot += 1
            prompt = render_prompt(row["problem"], system=system)
            completion = f"{row['solution']}\n\nANSWER: {row['answer']}{EOT}"
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "answer": row["answer"],
                    }
                )
                + "\n"
            )
    print(f"train -> {out_path} ({len(rows)} rows, {n_fewshot} with a few-shot prefix)")

    # plain-text dump for the contamination checker
    doc_path = os.path.join(args.out_dir, "sft_train_docs.jsonl")
    with open(doc_path, "w") as f:
        for row in rows:
            f.write(json.dumps({"text": row["problem"] + "\n" + row["solution"]}) + "\n")
    print(f"contamination input -> {doc_path}")


if __name__ == "__main__":
    main()
