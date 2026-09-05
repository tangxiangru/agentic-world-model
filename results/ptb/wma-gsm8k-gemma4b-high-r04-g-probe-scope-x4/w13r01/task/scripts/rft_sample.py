#!/usr/bin/env python3
"""Rejection-sampling data generation from a checkpoint of our own.

Samples k chains per training question with vLLM, keeps only those whose final
'ANSWER: <n>' line matches the gold answer, and writes them in exactly the same
render as scripts/build_sft.py so the SFT trainer can consume the file unchanged.

Questions come from the GSM8K *train* split and from OpenMathInstruct-2 problems
(themselves GSM8K/MATH-train-derived). The GSM8K test split is never read.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

from build_sft import MATH_PROMPT_TEMPLATE, fewshot_prefix, norm_answer

TEMPLATE_PATH = "templates/gemma3.jinja"
BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
ANSWER_TAIL = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def load_questions(args):
    """(question, gold) pairs, gold as a normalised integer string."""
    import pyarrow.parquet as pq
    from datasets import load_dataset

    seen, out = set(), []
    ds = load_dataset("openai/gsm8k", "main")["train"]
    for r in ds:
        a = norm_answer(r["answer"].split("####")[-1])
        q = r["question"].strip()
        if a and q not in seen:
            seen.add(q)
            out.append((q, a, "gsm8k_train"))
    n_gsm = len(out)

    if args.max_aug:
        shards = sorted(glob.glob(os.path.expanduser(
            "~/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet")))
        n = 0
        for p in shards:
            t = pq.read_table(p, columns=["problem", "expected_answer", "problem_source"])
            for r in t.to_pylist():
                if r["problem_source"] != "augmented_gsm8k":
                    continue
                a = norm_answer(r["expected_answer"] or "")
                q = (r["problem"] or "").strip()
                if not a or q in seen:
                    continue
                seen.add(q)
                out.append((q, a, "augmented_gsm8k"))
                n += 1
                if n >= args.max_aug:
                    break
            if n >= args.max_aug:
                break
    print(f"questions: {n_gsm} gsm8k_train + {len(out)-n_gsm} augmented_gsm8k "
          f"= {len(out)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--max-aug", type=int, default=12000)
    ap.add_argument("--keep-per-q", type=int, default=2)
    ap.add_argument("--hard-boost", type=int, default=1,
                    help="keep keep-per-q chains for questions solved by at most "
                         "half the samples, 1 chain for the easy rest")
    ap.add_argument("--fewshot-frac", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--limit-q", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(BASE)
    template = open(TEMPLATE_PATH).read()
    rng = random.Random(args.seed)

    qs = load_questions(args)
    if args.limit_q:
        qs = qs[: args.limit_q]

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            chat_template=template, tokenize=False, add_generation_prompt=True)
        for q, _, _ in qs
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=2048, dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_new, seed=args.seed,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept, stats = [], Counter()
    solved_any = 0
    for (q, gold, src), o in zip(qs, outs):
        good, seen_body = [], set()
        for c in o.outputs:
            txt = c.text.strip()
            stats["sampled"] += 1
            m = ANSWER_TAIL.search(txt)
            if not m:
                stats["no_answer_line"] += 1
                continue
            if norm_answer(m.group(1)) != gold:
                stats["wrong"] += 1
                continue
            if txt.count("ANSWER:") != 1:
                stats["multi_marker"] += 1
                continue
            key = hashlib.md5(re.sub(r"\s+", " ", txt).encode()).hexdigest()
            if key in seen_body:
                stats["dup"] += 1
                continue
            seen_body.add(key)
            good.append(txt)
        if good:
            solved_any += 1
        # Upweight the problems that carry signal. A question all k samples get
        # right teaches the model almost nothing it does not already do; one it
        # gets right only sometimes is where the correct chain is informative.
        solve_rate = len(good) / max(args.k, 1)
        cap = args.keep_per_q
        if args.hard_boost and solve_rate > 0.5:
            cap = 1
        stats["easy_q" if solve_rate > 0.5 else "hard_q"] += 1 if good else 0
        # prefer the shortest correct chains: they are the least likely to ramble
        good.sort(key=len)
        for txt in good[:cap]:
            kept.append((q, txt, gold, src))
            stats["kept"] += 1

    print("sampling stats", dict(stats), flush=True)
    print(f"questions with >=1 correct sample: {solved_any}/{len(qs)} "
          f"({solved_any/max(len(qs),1):.3f})", flush=True)

    prefix = fewshot_prefix()
    rng.shuffle(kept)
    rows, dropped = [], 0
    for q, body, gold, src in kept:
        msgs = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
        if rng.random() < args.fewshot_frac:
            msgs = [{"role": "system", "content": prefix}] + msgs
        prompt = tok.apply_chat_template(msgs, chat_template=template,
                                         tokenize=False, add_generation_prompt=True)
        completion = body + "<end_of_turn>"
        n_tok = len(tok(prompt + completion, add_special_tokens=False).input_ids)
        if n_tok > args.max_tokens:
            dropped += 1
            continue
        rows.append({"prompt": prompt, "completion": completion,
                     "text": prompt + completion, "source": "rft:" + src,
                     "answer": gold, "question": q, "n_tokens": n_tok})
    print(f"rows {len(rows)}; dropped {dropped} over {args.max_tokens}", flush=True)
    lens = sorted(r["n_tokens"] for r in rows)
    if lens:
        print(f"tokens p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} "
              f"max={lens[-1]} sum={sum(lens)/1e6:.2f}M", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "_check.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["question"] + "\n" + r["completion"]}) + "\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
