#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per training question
from one of my own checkpoints, keep only those whose ANSWER line matches the
reference answer, dedupe, and write a jsonl in the same prompt/completion shape
build_data.py produces.

Questions come from GSM8K *train* and from OpenMathInstruct-2's GSM8K-train-derived
augmented problems. No test item is ever read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

GSM8K_TRAIN = ("/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/"
               "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet")
CALC = re.compile(r"<<[^>]*>>")


def norm(a: str):
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(a)
    except ValueError:
        return None
    return round(f, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", required=True, help="jsonl from build_data.py, for augmented questions")
    ap.add_argument("--k-gold", type=int, default=6)
    ap.add_argument("--k-aug", type=int, default=2)
    ap.add_argument("--n-aug", type=int, default=20000)
    ap.add_argument("--max-per-question", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)

    tasks = []  # (question, gold_answer, k)
    for r in pq.ParquetFile(GSM8K_TRAIN).read().to_pylist():
        tasks.append((r["question"].strip(), r["answer"].rpartition("####")[2].strip(),
                      args.k_gold))

    seen = {q for q, _, _ in tasks}
    aug = []
    with open(args.pool) as f:
        for line in f:
            r = json.loads(line)
            if r["src"] == "augmented_gsm8k" and r["question"] not in seen:
                seen.add(r["question"])
                aug.append((r["question"], r["answer"], args.k_aug))
    rng.shuffle(aug)
    tasks += aug[: args.n_aug]
    print(f"questions: {len(tasks)} (gold x{args.k_gold} + aug x{args.k_aug})", flush=True)

    prompts, meta = [], []
    for q, a, k in tasks:
        p = render.render_prompt(tok, q)
        for _ in range(k):
            prompts.append(p)
            meta.append((q, a))
    print(f"generations: {len(prompts)}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, n=1,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept, per_q, seen_pair = [], {}, set()
    n_correct = 0
    for (q, gold), o in zip(meta, outs):
        text = o.outputs[0].text
        if o.outputs[0].finish_reason != "stop":
            continue
        m = re.search(r"ANSWER:\s*([^\n]*)", text)
        if not m:
            continue
        if norm(m.group(1)) is None or norm(m.group(1)) != norm(gold):
            continue
        n_correct += 1
        body = text[: m.end()].strip()
        if len(text[m.end():].strip()) > 2:
            continue  # did not terminate right after the answer
        if per_q.get(q, 0) >= args.max_per_question:
            continue
        key = (q, body)
        if key in seen_pair:
            continue
        seen_pair.add(key)
        per_q[q] = per_q.get(q, 0) + 1
        kept.append({"question": q, "answer": gold, "body": body})

    print(f"correct {n_correct}/{len(prompts)} = {n_correct/len(prompts):.3f}; "
          f"kept after dedup/cap: {len(kept)}; questions covered {len(per_q)}", flush=True)

    with open(args.out, "w") as f:
        for r in kept:
            prompt = render.render_prompt(tok, r["question"])
            completion = r["body"].strip() + render.STOP_TOKEN
            assert completion.count("ANSWER: ") == 1
            n = len(tok(prompt + completion, add_special_tokens=False)["input_ids"])
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "question": r["question"], "answer": r["answer"],
                                "src": "rft_self", "n_tokens": n}) + "\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
