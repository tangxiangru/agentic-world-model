#!/usr/bin/env python3
"""Second-stage SFT corpus: OpenMathInstruct-2 gsm8k-family solutions for the
problems that sft_v1.jsonl never used. Same formatting contract as sft_v1
(one 'ANSWER: <n>' line, <end_of_turn> terminator, some rows k-shot prefixed).

Difference from build_data.py: the k-shot demonstration blocks keep gsm8k's
<<48/2=24>> calculator annotations, because the grader's real 10-shot system
message keeps them and sft_v1's stripped demos were the one place training and
grading strings differed.
"""
import argparse
import glob
import json
import random

import pandas as pd

from build_data import (MATH_PROMPT_TEMPLATE, OMI2_GLOB, STOP, make_row,
                        norm_answer, strip_boxed)


def used_questions(path: str) -> set:
    used = set()
    for line in open(path):
        p = json.loads(line)["prompt"]
        i = p.rfind("Solve the following math problem step by step.")
        used.add(p[i:].split("\n\n")[1].strip())
    return used


def gsm8k_demos():
    """Demonstrations in the harness's exact few-shot format, annotations kept."""
    from datasets import load_dataset

    out = []
    for r in load_dataset("openai/gsm8k", "main", split="train"):
        parts = r["answer"].split("####")
        ans = norm_answer(parts[-1])
        if ans is None:
            continue
        out.append({"question": r["question"].strip(),
                    "solution": "####".join(parts[:-1]).strip(), "answer": ans})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-v1", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--n-max", type=int, default=58000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="/home/ben/task/data/sft_v2_fresh.jsonl")
    ap.add_argument("--decon-out", default="/home/ben/task/data/sft_v2_fresh.decon.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    used = used_questions(args.sft_v1)
    print(f"problems already used by sft_v1: {len(used)}")

    frames = []
    for f in sorted(glob.glob(OMI2_GLOB)):
        df = pd.read_parquet(f, columns=["problem", "generated_solution",
                                         "expected_answer", "problem_source"])
        frames.append(df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])])
    df = pd.concat(frames, ignore_index=True)

    idx = list(range(len(df)))
    rng.shuffle(idx)
    seen, pool = {}, []
    for i in idx:
        r = df.iloc[i]
        q = r.problem.strip()
        if q in used or seen.get(q, 0) >= args.max_per_problem:
            continue
        ans = norm_answer(str(r.expected_answer))
        if ans is None:
            continue
        sol = strip_boxed(r.generated_solution)
        if not sol or len(sol) < 30:
            continue
        row = make_row(q, sol, ans)
        if row is None:
            continue
        seen[q] = seen.get(q, 0) + 1
        row["src"] = r.problem_source
        pool.append(row)
    print(f"fresh rows available: {len(pool)} over {len(seen)} unseen problems")
    rng.shuffle(pool)
    pool = pool[: args.n_max]

    demos = gsm8k_demos()
    n_fs = int(len(pool) * args.fewshot_frac)
    with open(args.out, "w") as f, open(args.decon_out, "w") as g:
        for i, row in enumerate(pool):
            user = MATH_PROMPT_TEMPLATE.format(prompt=row["question"])
            if i < n_fs:
                k = rng.choice([2, 3, 4, 6, 8])
                block = "\n\n".join(
                    f"{p['question']}\n\nReasoning:\n{p['solution']}\n\nANSWER: {p['answer']}"
                    for p in rng.sample(demos, k)
                )
                user = block + "\n\n" + user
            f.write(json.dumps({"prompt": user, "completion": row["completion"],
                                "src": row["src"]}) + "\n")
            g.write(json.dumps({"question": row["question"],
                                "answer": row["completion"][: -len(STOP)]}) + "\n")
    from collections import Counter
    print(f"wrote {len(pool)} rows -> {args.out}  (fewshot-prefixed {n_fs})")
    print(" ", Counter(r["src"] for r in pool))


if __name__ == "__main__":
    main()
