"""Build the SFT mixture for GSM8K.

Sources (all train-split / train-derived, never the GSM8K test split):
  * openai/gsm8k `train` split, original human CoT reformatted to the eval format.
  * nvidia/OpenMathInstruct-2 rows whose problem_source is gsm8k / augmented_gsm8k
    (problems seeded from the GSM8K *train* split, solutions from Llama-3.1-405B),
    plus a small slice of augmented_math for numeric breadth.

Output: data/sft_pool.jsonl with {question, solution, answer, source}.
"""
import argparse
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

from common import normalize_num

OMI_DIR = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data"

CALC_RE = re.compile(r"<<[^>]*>>")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def clean_boxed(text: str) -> str:
    text = BOXED_RE.sub(r"\1", text)
    text = text.replace("\\boxed", "")
    return text


def ok_solution(sol: str) -> bool:
    if not sol or len(sol) < 20 or len(sol) > 4000:
        return False
    # drop degenerate / self-contradicting generations
    low = sol.lower()
    if "i apologize" in low or "wait, that" in low:
        return False
    return True


def build_gsm8k_train():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        q = r["question"].strip()
        parts = r["answer"].split("####")
        target = normalize_num(parts[-1].strip())
        if target is None:
            continue
        reasoning = CALC_RE.sub("", "####".join(parts[:-1])).strip()
        reasoning = re.sub(r"[ \t]+", " ", reasoning)
        sol = f"{reasoning}\n\nANSWER: {target}"
        out.append(dict(question=q, solution=sol, answer=target, source="gsm8k_orig"))
    return out


def build_omi(n_gsm8k, n_aug_gsm8k, n_aug_math, seed=0, skip=(0, 0, 0)):
    rng = random.Random(seed)
    buckets = {"gsm8k": [], "augmented_gsm8k": [], "augmented_math": []}
    files = sorted(glob.glob(os.path.join(OMI_DIR, "*.parquet")))
    print(f"reading {len(files)} OMI-2 shards")
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=50000):
            d = batch.to_pydict()
            for prob, sol, ans, src in zip(
                d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]
            ):
                if src not in buckets:
                    continue
                a = normalize_num(ans)
                if a is None:
                    continue
                if not ok_solution(sol):
                    continue
                sol = clean_boxed(sol).strip()
                if src == "augmented_math" and ("\\frac" in sol or "\\sqrt" in sol):
                    # keep the math slice simple / grade-school-ish
                    continue
                buckets[src].append(
                    dict(question=prob.strip(), solution=f"{sol}\n\nANSWER: {a}", answer=a, source=src)
                )
    for k, v in buckets.items():
        print(f"  {k}: {len(v)}")
    out = []
    for (k, n), sk in zip(
        (("gsm8k", n_gsm8k), ("augmented_gsm8k", n_aug_gsm8k), ("augmented_math", n_aug_math)), skip
    ):
        v = buckets[k]
        rng.shuffle(v)
        out.extend(v[sk : sk + n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_pool.jsonl")
    ap.add_argument("--n-gsm8k", type=int, default=30000)
    ap.add_argument("--n-aug-gsm8k", type=int, default=110000)
    ap.add_argument("--n-aug-math", type=int, default=12000)
    ap.add_argument("--gsm8k-orig-repeat", type=int, default=2)
    ap.add_argument("--skip", type=int, nargs=3, default=[0, 0, 0])
    args = ap.parse_args()

    rows = []
    orig = build_gsm8k_train()
    print("gsm8k_orig:", len(orig))
    for _ in range(args.gsm8k_orig_repeat):
        rows.extend(orig)
    rows.extend(build_omi(args.n_gsm8k, args.n_aug_gsm8k, args.n_aug_math, skip=tuple(args.skip)))

    random.Random(1234).shuffle(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("total rows:", len(rows), "->", args.out)


if __name__ == "__main__":
    main()
