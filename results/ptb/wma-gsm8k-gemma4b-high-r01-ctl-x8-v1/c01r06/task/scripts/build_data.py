"""Build the SFT file: prompt / completion pairs in the grader's exact format.

Sources (all GSM8K *train*-derived or train-split; the GSM8K test split is
never read here):
  * openai/gsm8k main/train  - original human solutions
  * nvidia/OpenMathInstruct-2, rows with problem_source in
    {gsm8k, augmented_gsm8k} - Llama-3.1-405B-Instruct solutions to GSM8K train
    problems and to augmentations of them, answer-verified by the dataset authors
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

GSM8K_TRAIN = glob.glob(
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
)
OMI2 = sorted(
    glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-*.parquet"
    )
)


def load_gsm8k_train():
    rows = []
    for f in GSM8K_TRAIN:
        d = pq.read_table(f).to_pydict()
        for q, a in zip(d["question"], d["answer"]):
            ans = a.split("####")[-1].strip()
            reasoning = fmt.normalize_solution(a.split("####")[0])
            rows.append({"question": q, "reasoning": reasoning, "answer": ans})
    return rows


def load_omi2(max_per_problem, sources):
    by_problem = {}
    for f in OMI2:
        d = pq.read_table(f, columns=["problem", "generated_solution",
                                      "expected_answer", "problem_source"]).to_pydict()
        for p, s, a, src in zip(d["problem"], d["generated_solution"],
                                d["expected_answer"], d["problem_source"]):
            if src not in sources:
                continue
            t = fmt.build_target(s, a)
            if t is None:
                continue
            lst = by_problem.setdefault(p, [])
            if len(lst) < max_per_problem and t not in lst:
                lst.append(t)
    return by_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default="gsm8k,augmented_gsm8k")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm = load_gsm8k_train()
    print(f"gsm8k train rows: {len(gsm)}", flush=True)

    examples = []  # (question, target)
    for r in gsm:
        t = fmt.build_target(r["reasoning"], r["answer"])
        if t:
            examples.append((r["question"], t))
    n_orig = len(examples)

    by_problem = load_omi2(args.max_per_problem, set(args.sources.split(",")))
    print(f"omi2 unique problems: {len(by_problem)}", flush=True)
    for p, ts in by_problem.items():
        for t in ts:
            examples.append((p, t))
    print(f"total candidate rows: {len(examples)} (orig gsm8k {n_orig})", flush=True)

    rng.shuffle(examples)
    if args.max_rows:
        examples = examples[: args.max_rows]

    # pool of few-shot exemplars, drawn only from gsm8k train, rendered exactly
    # as inspect_evals/gsm8k::sample_to_fewshot does
    shot_pool = [
        fmt.fewshot_block(r["question"], r["reasoning"], r["answer"]) for r in gsm
    ]

    n_bad = 0
    with open(args.out, "w") as fh:
        for q, t in examples:
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 4, 10])
                system = "\n\n".join(rng.sample(shot_pool, k))
            prompt = fmt.render_prompt(q, system)
            completion = fmt.render_completion(t)
            if completion.count("ANSWER:") != 1 or not completion.endswith(fmt.STOP_TOKEN):
                n_bad += 1
                continue
            fh.write(json.dumps({"prompt": prompt, "completion": completion,
                                 "n_shot": 0 if system is None else k}) + "\n")
    print(f"wrote {args.out}; dropped {n_bad}", flush=True)


if __name__ == "__main__":
    main()
