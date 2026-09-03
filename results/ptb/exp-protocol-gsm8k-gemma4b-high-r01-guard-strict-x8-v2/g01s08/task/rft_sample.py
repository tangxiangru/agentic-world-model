#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per training problem from a checkpoint,
keep the ones whose final ANSWER matches gold, and write SFT rows.

Problem pool: GSM8K TRAIN questions and the GSM8K-derived augmented problems of
OpenMathInstruct-2 (never the test split).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
SHARDS = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def norm(s):
    return s.replace(",", "").replace("$", "").strip().rstrip(".")


def as_float(s):
    try:
        return float(norm(s))
    except ValueError:
        return None


def problem_pool(n_aug, seed):
    pool = []
    gsm = load_dataset("openai/gsm8k", "main")["train"]
    for r in gsm:
        pool.append((r["question"].strip(), norm(r["answer"].split("####")[-1]), "gsm8k_train"))
    seen = {q for q, _, _ in pool}
    aug = []
    for f in sorted(glob.glob(SHARDS)):
        t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pylist()
        for r in t:
            if r["problem_source"] != "augmented_gsm8k":
                continue
            q = r["problem"].strip()
            a = norm(r["expected_answer"] or "")
            if q in seen or not NUMLIKE.match(a):
                continue
            seen.add(q)
            aug.append((q, a, "augmented_gsm8k"))
        if len(aug) > n_aug * 3:
            break
    random.Random(seed).shuffle(aug)
    return pool + aug[:n_aug]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--k-gsm8k", type=int, default=8)
    ap.add_argument("--n-aug", type=int, default=30000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = open(TEMPLATE).read()

    pool = problem_pool(args.n_aug, args.seed)
    print(f"problems: {len(pool)}", flush=True)

    prompts, meta = [], []
    for q, a, src in pool:
        p = tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            tokenize=False, add_generation_prompt=True,
        )
        prompts.append(p)
        meta.append((q, a, src))

    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=1280,
              dtype="bfloat16", enable_prefix_caching=True, seed=args.seed)

    def run(idx, k):
        sp = SamplingParams(n=k, temperature=args.temp, top_p=0.95,
                            max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
        outs = llm.generate([prompts[i] for i in idx], sp)
        return outs

    idx_gsm = [i for i, m in enumerate(meta) if m[2] == "gsm8k_train"]
    idx_aug = [i for i, m in enumerate(meta) if m[2] != "gsm8k_train"]

    rows, stats = [], {"n_problems": 0, "n_samples": 0, "n_correct": 0, "solved_any": 0}
    for idx, k in ((idx_gsm, args.k_gsm8k), (idx_aug, args.k)):
        if not idx:
            continue
        outs = run(idx, k)
        for i, o in zip(idx, outs):
            q, gold, src = meta[i]
            g = as_float(gold)
            kept, seen_txt = 0, set()
            stats["n_problems"] += 1
            any_ok = False
            for c in o.outputs:
                stats["n_samples"] += 1
                txt = c.text.strip()
                m = ANS.findall(txt)
                if len(m) != 1:
                    continue
                v = as_float(m[0])
                if v is None or g is None or abs(v - g) > 1e-6:
                    continue
                stats["n_correct"] += 1
                any_ok = True
                if kept >= args.keep_per_problem or txt in seen_txt:
                    continue
                # normalise the ending: exactly one ANSWER line, then the stop token
                body = txt[: txt.rindex("ANSWER:")].rstrip()
                completion = f"{body}\n\nANSWER: {gold}<end_of_turn>"
                seen_txt.add(txt)
                kept += 1
                rows.append({"question": q, "answer": f"{body}\n#### {gold}",
                             "prompt": prompts[i], "completion": completion, "source": src})
            stats["solved_any"] += int(any_ok)

    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    stats["rows_written"] = len(rows)
    stats["pass_rate"] = stats["n_correct"] / max(1, stats["n_samples"])
    stats["solve_any_rate"] = stats["solved_any"] / max(1, stats["n_problems"])
    print(json.dumps(stats, indent=2))
    if args.stats:
        json.dump(stats, open(args.stats, "w"), indent=2)


if __name__ == "__main__":
    main()
