#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per problem from a
checkpoint, keep the ones whose final number equals gold, write SFT rows.

Prompts are rendered with the grader's chat template so the sampled solutions
are on-policy for exactly the distribution the model is graded on.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pandas as pd
from transformers import AutoTokenizer

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def final_number(text: str) -> str | None:
    nums = NUM.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def load_problems(n_aug: int, seed: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    df = pd.read_parquet(sorted(glob.glob(GSM8K_TRAIN))[0])
    for _, r in df.iterrows():
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append((r["question"].strip(), ans))
    if n_aug:
        frames = []
        for f in sorted(glob.glob(OMI2)):
            d = pd.read_parquet(f, columns=["problem", "expected_answer", "problem_source"])
            frames.append(d[d["problem_source"] == "augmented_gsm8k"])
        d = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["problem"])
        d = d[d["expected_answer"].astype(str).str.match(r"^-?\d{1,12}$")]
        rng = random.Random(seed)
        idx = rng.sample(range(len(d)), min(n_aug, len(d)))
        for i in idx:
            r = d.iloc[i]
            out.append((r["problem"].strip(), str(r["expected_answer"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-aug", type=int, default=10000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--fewshot-file", default="data/fewshot_system.txt")
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--max-problems", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    problems = load_problems(args.n_aug, args.seed)
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"{len(problems)} problems")

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open("templates/gemma3.jinja").read()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            chat_template=template, tokenize=False, add_generation_prompt=True,
        )
        for q, _ in problems
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, dtype="bfloat16", seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    fewshot = open(args.fewshot_file).read().strip()
    rng = random.Random(args.seed + 7)
    rows = []
    n_solved = 0
    for (q, gold), o in zip(problems, outs):
        kept = []
        seen = set()
        for c in o.outputs:
            t = c.text.strip()
            if final_number(t) != gold:
                continue
            # keep distinct derivations only
            key = re.sub(r"\s+", " ", t)[:120]
            if key in seen:
                continue
            seen.add(key)
            kept.append(t)
            if len(kept) >= args.keep_per_problem:
                break
        if kept:
            n_solved += 1
        for t in kept:
            rows.append({"question": q, "target": t})
    print(f"{n_solved}/{len(problems)} problems solved at least once; {len(rows)} rows kept")

    rng.shuffle(rows)
    n_fs = int(len(rows) * args.fewshot_frac)
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "system": fewshot if i < n_fs else None,
                "user": MATH_PROMPT_TEMPLATE.format(prompt=r["question"]),
                "completion": r["target"] + "<end_of_turn>",
                "source": "rft:self",
            }) + "\n")
    if args.stats_out:
        json.dump({"problems": len(problems), "solved": n_solved, "rows": len(rows),
                   "k": args.k, "temperature": args.temperature},
                  open(args.stats_out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
