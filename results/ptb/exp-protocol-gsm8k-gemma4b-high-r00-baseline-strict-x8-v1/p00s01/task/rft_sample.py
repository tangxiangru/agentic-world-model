#!/usr/bin/env python3
"""Rejection-sampling data generation from my own SFT checkpoint.

For each problem (GSM8K-train-derived, gold answer known, never a test item),
sample k completions with the grader's own prompt, keep the ones whose graded
answer equals gold, and emit them as new SFT rows in the same format.

Problems where all k samples are already correct are the ones the model has
mastered; problems where none are correct are out of reach for now. The default
filter keeps the middle band, which is where the training signal is.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from build_data import MATH_PROMPT_TEMPLATE, STOP, render, sample_to_fewshot
from dev_eval import graded_answer, norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sources", nargs="+", default=["data/sft_omi2_gsm8k.jsonl"])
    ap.add_argument("--n-problems", type=int, default=24000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--min-correct", type=int, default=1)
    ap.add_argument("--max-correct", type=int, default=3)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    seen = set()
    problems = []
    for src in args.sources:
        with open(src) as f:
            for line in f:
                r = json.loads(line)
                q = r["question"]
                if q in seen:
                    continue
                m = re.search(r"ANSWER: ([^\n<]+)", r["completion"])
                if not m:
                    continue
                seen.add(q)
                problems.append((q, m.group(1).strip()))
    rng.shuffle(problems)
    problems = problems[: args.n_problems]
    print(f"sampling {args.k} completions for {len(problems)} problems", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    prompts = [render(None, MATH_PROMPT_TEMPLATE.format(prompt=q)) for q, _ in problems]
    outs = llm.generate(prompts, sp)

    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    demos = []
    for r in train:
        parts = r["answer"].split("####")
        demos.append((r["question"], "####".join(parts[:-1]).strip(), parts[-1].strip()))

    hist = Counter()
    n_written = 0
    with open(args.out, "w") as f:
        for (q, gold), o in zip(problems, outs):
            g = norm(gold)
            good = []
            for c in o.outputs:
                if c.finish_reason == "length":
                    continue
                if graded_answer(c.text) == g:
                    good.append(c.text.strip())
            hist[len(good)] += 1
            if not (args.min_correct <= len(good) <= args.max_correct):
                continue
            # dedup identical solutions, prefer the shortest (least padding, fewest detours)
            uniq = sorted(set(good), key=len)
            for text in uniq[: args.keep_per_problem]:
                if "ANSWER:" not in text:
                    continue
                comp = text + STOP
                if comp.count("ANSWER:") != 1:
                    continue
                system = None
                if rng.random() < args.fewshot_frac:
                    k = rng.randint(2, 10)
                    system = "\n\n".join(sample_to_fewshot(*d) for d in rng.sample(demos, k))
                prompt = render(system, MATH_PROMPT_TEMPLATE.format(prompt=q))
                f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "completion": comp,
                            "question": q,
                            "answer": text,
                        }
                    )
                    + "\n"
                )
                n_written += 1

    stats = {
        "problems": len(problems),
        "k": args.k,
        "n_correct_histogram": {str(i): hist[i] for i in range(args.k + 1)},
        "pass_at_1_est": sum(i * hist[i] for i in hist) / (args.k * len(problems)),
        "rows_written": n_written,
    }
    print(json.dumps(stats, indent=1))
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
