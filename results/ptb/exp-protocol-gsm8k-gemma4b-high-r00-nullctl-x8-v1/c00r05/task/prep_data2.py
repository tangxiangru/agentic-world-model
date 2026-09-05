#!/usr/bin/env python3
"""Stage-3 dataset: fresh OpenMathInstruct-2 problems (not used in sft_v1) + on-policy RFT data."""
import argparse, glob, json, random, os
import pandas as pd
from prep_data import (MATH_PROMPT_TEMPLATE, clean_solution, is_int_answer,
                       norm_int, sample_to_fewshot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--used", default="data/sft_v1.jsonl")
    ap.add_argument("--rft", default="data/rft_sft.jsonl")
    ap.add_argument("--out", default="data/sft_v2.jsonl")
    ap.add_argument("--n-gsm", type=int, default=90000)
    ap.add_argument("--n-math", type=int, default=20000)
    ap.add_argument("--n-rft", type=int, default=45000)
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    used_problems = set()
    with open(args.used) as f:
        for line in f:
            used_problems.add(json.loads(line)["problem"])
    print("used problems:", len(used_problems))

    from datasets import load_dataset
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    fs_pool = []
    for r in gsm_train:
        parts = r["answer"].split("####")
        fs_pool.append((r["question"], "####".join(parts[:-1]).strip(),
                        parts[-1].strip()))

    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    print("shards", len(files))
    per_problem = {}
    gsm_rows, math_rows = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["problem", "generated_solution",
                                         "expected_answer", "problem_source"])
        for prob, sol, ans, src in df.itertuples(index=False):
            if prob in used_problems or not is_int_answer(ans):
                continue
            if len(sol) > 3000 or len(sol) < 40:
                continue
            if src in ("gsm8k", "augmented_gsm8k"):
                bucket = gsm_rows
            elif src in ("math", "augmented_math"):
                bucket = math_rows
            else:
                continue
            c = per_problem.get(prob, 0)
            if c >= args.max_per_problem:
                continue
            per_problem[prob] = c + 1
            bucket.append((prob, sol, norm_int(ans)))
        print(f.split("/")[-1], len(gsm_rows), len(math_rows), flush=True)
        if len(gsm_rows) > args.n_gsm * 1.5 and len(math_rows) > args.n_math * 1.5:
            break

    rng.shuffle(gsm_rows)
    rng.shuffle(math_rows)
    rows = []
    for prob, sol, ans in gsm_rows[:args.n_gsm] + math_rows[:args.n_math]:
        rows.append((prob, f"{clean_solution(sol)}\n\nANSWER: {ans}"))
    print("fresh omi rows", len(rows))

    rft = []
    with open(args.rft) as f:
        for line in f:
            d = json.loads(line)
            rft.append((d["problem"], d["response"]))
    rng.shuffle(rft)
    rft = rft[:args.n_rft]
    print("rft rows", len(rft))
    rows.extend(rft)
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fo:
        for prob, resp in rows:
            user = MATH_PROMPT_TEMPLATE.format(prompt=prob.strip())
            if rng.random() < args.fewshot_frac:
                shots = rng.sample(fs_pool, rng.randint(1, 10))
                block = "\n\n".join(sample_to_fewshot(*s) for s in shots)
                user = block + "\n\n" + user
            fo.write(json.dumps({"prompt": user, "response": resp,
                                 "problem": prob}) + "\n")
    print("wrote", args.out, len(rows))


if __name__ == "__main__":
    main()
