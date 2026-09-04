"""Build the SFT file.

Sources (all GSM8K *train*-derived, no test items anywhere):
  * nvidia/OpenMathInstruct-2, rows with problem_source in {gsm8k, augmented_gsm8k}
    (405B-generated solutions to GSM8K train problems and to augmentations of them)
  * openai/gsm8k main/train, its own reference solutions with the <<..>> calculator
    annotations stripped

Every target is rendered with gsm_format so it ends on <end_of_turn> and carries
exactly one "ANSWER: " line, whose number is the last numeric token in the string.

Writes:
  data/sft_train.jsonl        {prompt, completion}   - what the trainer reads
  data/sft_contam.jsonl       {question, answer}     - what the checker reads
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pandas as pd
import pyarrow.parquet as pq

import gsm_format as G

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"

NUM_RE = re.compile(r"^-?\d{1,12}(\.\d{1,4})?$")
CALC_RE = re.compile(r"<<[^>]*>>")


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "")
    if not NUM_RE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def omi2_rows(max_per_problem: int, cap: int, rng: random.Random):
    """gsm8k-sourced OpenMathInstruct-2 rows, boxed answer rewritten to ANSWER:."""
    per_problem: dict[str, int] = {}
    out = []
    files = sorted(glob.glob(OMI2))
    for f in files:
        df = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"]).to_pandas()
        df = df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])]
        for problem, sol, ans, _src in df.itertuples(index=False):
            ans = clean_answer(ans)
            if ans is None:
                continue
            if sol.count("\\boxed") != 1:
                continue
            head, _, _tail = sol.partition("\\boxed")
            # drop the partial sentence/line the boxed answer was sitting in
            head = head.rstrip()
            cut = max(head.rfind("\n"), head.rfind(". "))
            if cut <= 0:
                continue
            reasoning = head[: cut + 1].strip() if head[cut] == "." else head[:cut].strip()
            if len(reasoning) < 40 or "\\boxed" in reasoning:
                continue
            if per_problem.get(problem, 0) >= max_per_problem:
                continue
            per_problem[problem] = per_problem.get(problem, 0) + 1
            out.append((problem, reasoning, ans))
        print(f"  {f.split('/')[-1]}: kept {len(out)} so far", flush=True)
        if len(out) >= cap * 2:
            break
    rng.shuffle(out)
    return out[:cap]


def gsm8k_train_rows():
    df = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(GSM8K_TRAIN))])
    out = []
    for q, a in zip(df.question, df.answer):
        body, _, final = a.partition("####")
        ans = clean_answer(final)
        if ans is None:
            continue
        reasoning = CALC_RE.sub("", body).strip()
        out.append((q, reasoning, ans))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-omi2", type=int, default=60000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sft_train.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("reading OpenMathInstruct-2 ...", flush=True)
    rows = omi2_rows(args.max_per_problem, args.cap_omi2, rng)
    print("omi2 rows:", len(rows))

    native = gsm8k_train_rows()
    print("gsm8k train rows:", len(native))
    rows += native * args.gsm8k_repeat

    # few-shot prefixes are drawn from the gsm8k TRAIN reference solutions, the
    # same pool the grader's 10-shot system message comes from.
    shot_pool = native
    rng.shuffle(rows)

    n_fs = 0
    with open(args.out, "w") as f, open("data/sft_contam.jsonl", "w") as fc:
        for question, reasoning, ans in rows:
            shots = None
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 4, 6, 8, 10])
                shots = rng.sample(shot_pool, k)
                n_fs += 1
            f.write(json.dumps({
                "prompt": G.render_prompt(question, fewshots=shots),
                "completion": G.render_target(reasoning, ans),
            }) + "\n")
            fc.write(json.dumps({"question": question, "answer": f"{reasoning}\nANSWER: {ans}"}) + "\n")

    print(f"wrote {len(rows)} rows to {args.out} ({n_fs} with a few-shot prefix)")


if __name__ == "__main__":
    main()
