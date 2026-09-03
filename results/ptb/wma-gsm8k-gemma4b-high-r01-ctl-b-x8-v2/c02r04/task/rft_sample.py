#!/usr/bin/env python3
"""Rejection-sampling data from the model's own generations.

Question pool: openai/gsm8k main/TRAIN plus OpenMathInstruct-2 gsm8k-derived
problems (both carry a gold answer). The GSM8K test split is never touched.

For each question we draw k samples at temperature T, keep the ones whose final
"ANSWER: n" matches gold, and then keep only questions in a pass-rate band -
questions the model already solves every time teach it nothing, and questions it
never solves give no correct chain to learn from. What is left is the frontier.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from build_data import (MATH_PROMPT_TEMPLATE, NUM_RE, clean_solution,
                        sample_to_fewshot)

GSM8K_TRAIN = glob.glob(
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet")[0]
OMI2_FULL = sorted(glob.glob(
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-000*.parquet"))
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def norm(x: str) -> str | None:
    try:
        return f"{float(x.replace(',', '')):.6f}"
    except ValueError:
        return None


def build_pool(n_aug: int, seed: int):
    rows = pq.read_table(GSM8K_TRAIN).to_pylist()
    pool = [(r["question"].strip(), r["answer"].split("####")[-1].strip())
            for r in rows]
    fewshot = [sample_to_fewshot(r["question"], r["answer"]) for r in rows]
    if n_aug:
        seen = {q for q, _ in pool}
        aug = []
        for f in OMI2_FULL:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=20000):
                for r in batch.to_pylist():
                    if r["problem_source"] != "augmented_gsm8k":
                        continue
                    a = (r["expected_answer"] or "").strip()
                    q = r["problem"].strip()
                    if not NUM_RE.match(a) or q in seen:
                        continue
                    seen.add(q)
                    aug.append((q, a))
            if len(aug) >= n_aug * 3:
                break
        random.Random(seed).shuffle(aug)
        pool += aug[:n_aug]
    return pool, fewshot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft_v1.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-keep", type=int, default=2)
    ap.add_argument("--n-aug", type=int, default=20000)
    ap.add_argument("--pass-rate-max", type=float, default=0.75,
                    help="drop questions the model already solves this often")
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool, fewshot_pool = build_pool(args.n_aug, args.seed)
    print(f"question pool: {len(pool)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(TEMPLATE).read()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            tokenize=False, add_generation_prompt=True)
        for q, _ in pool
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=3072,
              dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106],
                        seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept, hist = [], defaultdict(int)
    n_any = n_band = 0
    for (q, gold), out in zip(pool, outs):
        gnorm = norm(gold)
        cands, seen = [], set()
        for c in out.outputs:
            txt = c.text.strip()
            m = ANS_RE.search(txt)
            if not m or norm(m.group(1)) != gnorm:
                continue
            body = txt[: m.start()].rstrip()
            if len(body) < 20 or "ANSWER:" in body or "boxed" in txt or "####" in txt:
                continue
            clean = f"{body}\n\nANSWER: {gold}"
            if clean[:160] in seen:
                continue
            seen.add(clean[:160])
            cands.append(clean)
        rate = len(cands) / args.k
        hist[round(rate, 2)] += 1
        if cands:
            n_any += 1
        if not cands or rate > args.pass_rate_max:
            continue
        n_band += 1
        cands.sort(key=len)
        for clean in cands[: args.max_keep]:
            messages = []
            if rng.random() < args.fewshot_frac:
                kk = rng.choice([2, 3, 4, 6, 10])
                messages.append({"role": "system",
                                 "content": "\n\n".join(rng.sample(fewshot_pool, kk))})
            messages.append({"role": "user",
                             "content": MATH_PROMPT_TEMPLATE.format(prompt=q)})
            messages.append({"role": "assistant", "content": clean})
            kept.append({"messages": messages, "completion": clean,
                         "question": q, "answer": gold, "source": "rft:self"})

    rng.shuffle(kept)
    with open(args.out, "w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    stats = {"questions": len(pool), "k": args.k,
             "questions_with_at_least_one_correct": n_any,
             "pass_at_k": round(n_any / len(pool), 4),
             "questions_in_band": n_band,
             "pass_rate_histogram": {str(k): v for k, v in sorted(hist.items())},
             "rows_written": len(kept)}
    json.dump(stats, open(args.out.replace(".jsonl", "_stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
