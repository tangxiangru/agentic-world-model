#!/usr/bin/env python3
"""Rejection-sampling data generation.

Samples k solutions per GSM8K *train* question from a checkpoint, keeps the ones
whose graded last number equals the gold answer, and writes them in the same
prompt/completion shape build_data.py produces.

Only the openai/gsm8k TRAIN split is touched. The benchmark's test split is
never read here.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re

import pyarrow.parquet as pq

from build_data import MATH_PROMPT_TEMPLATE, EOT, render_prompt, last_number, build_fewshot_pool, fewshot_block

GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--n-questions", type=int, default=0, help="0 = all")
    ap.add_argument("--fewshot-frac", type=float, default=0.20)
    ap.add_argument("--fewshot-max-k", type=int, default=10)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--pool", default=None, help="jsonl of {question,gold} to use instead of the gsm8k train split")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.pool:
        items = []
        for line in open(args.pool):
            d = json.loads(line)
            items.append((d["question"].strip(), d["gold"].strip().replace(",", "")))
        if args.n_questions:
            items = items[: args.n_questions]
    else:
        rows = pq.read_table(sorted(glob.glob(GSM8K_TRAIN))[0]).to_pylist()
        if args.n_questions:
            rows = rows[: args.n_questions]
        items = []
        for r in rows:
            q = r["question"].strip()
            gold = r["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
            items.append((q, gold))
    print(f"[rft] {len(items)} train questions")

    prompts = [render_prompt(None, MATH_PROMPT_TEMPLATE.format(prompt=q)) for q, _ in items]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=4096,
        dtype="bfloat16",
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=None,
    )
    outs = llm.generate(prompts, sp)

    pool = build_fewshot_pool()
    kept, n_correct, n_total, solved_q = [], 0, 0, 0
    for (q, gold), o in zip(items, outs):
        seen = set()
        good = []
        for c in o.outputs:
            n_total += 1
            txt = c.text.strip()
            if not txt:
                continue
            if last_number(txt) != last_number("x " + gold):
                continue
            if txt.count("ANSWER:") != 1:
                continue
            if not re.search(r"ANSWER:\s*\$?-?[\d,]+(\.\d+)?\$?\s*$", txt):
                continue
            n_correct += 1
            h = hashlib.md5(re.sub(r"\s+", " ", txt).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            good.append(txt)
        if good:
            solved_q += 1
        rng.shuffle(good)
        for txt in good[: args.max_per_question]:
            kept.append((q, txt))

    rng.shuffle(kept)
    n_fs = 0
    with open(args.out, "w") as fh:
        for q, txt in kept:
            system = None
            if rng.random() < args.fewshot_frac:
                system = fewshot_block(pool, rng, rng.randint(1, args.fewshot_max_k))
                n_fs += 1
            fh.write(json.dumps({
                "prompt": render_prompt(system, MATH_PROMPT_TEMPLATE.format(prompt=q)),
                "completion": txt + EOT,
                "question": q,
                "answer": txt,
                "src": "rft",
                "fewshot": system is not None,
            }) + "\n")

    stats = {
        "model": args.model,
        "questions": len(items),
        "samples": n_total,
        "correct_samples": n_correct,
        "pass_rate": n_correct / max(n_total, 1),
        "questions_solved_at_least_once": solved_q,
        "solve_rate": solved_q / max(len(items), 1),
        "rows_written": len(kept),
        "rows_with_fewshot": n_fs,
    }
    print(json.dumps(stats, indent=2))
    if args.stats:
        json.dump(stats, open(args.stats, "w"), indent=2)


if __name__ == "__main__":
    main()
