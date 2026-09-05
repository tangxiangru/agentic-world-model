#!/usr/bin/env python3
"""Rejection-sampling data generation from a fine-tuned checkpoint.

Samples k solutions per problem with vLLM using the grader's own prompt format,
keeps the ones whose final 'ANSWER: <n>' line matches the reference answer, and
writes them back out in the same {prompt, completion} shape as build_data.py.
Problems come from OpenMathInstruct-2's gsm8k/augmented_gsm8k rows (train-derived
only); nothing from the benchmark test set is read.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer

TASK = "/home/ben/task"
BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
OMI = ("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
       "469216e3f46f4dacf476b382e192485ea51a143e/data")

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANSLINE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def norm(a: str):
    a = a.strip().replace(",", "").replace("$", "")
    try:
        v = float(a)
    except ValueError:
        return None
    return round(v, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=40000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-keep-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--skip-file", default=None,
                    help="jsonl of already-used rows; their problems are excluded")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open(f"{TASK}/templates/gemma3.jinja").read()
    sysmsg = open(f"{TASK}/data/fewshot_system.txt").read()

    skip = set()
    if args.skip_file:
        for line in open(args.skip_file):
            r = json.loads(line)
            p = r["prompt"].split("$ANSWER is the answer to the problem.\n\n", 1)[1]
            q = p.split("\n\nRemember to put your answer on its own line", 1)[0]
            skip.add(hashlib.md5(q.strip().encode()).hexdigest())
    print(f"skipping {len(skip)} already-used problems", flush=True)

    probs = []
    seen = set()
    for fp in sorted(glob.glob(f"{OMI}/train-*.parquet")):
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["problem", "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                a = norm(r["expected_answer"])
                if a is None:
                    continue
                q = r["problem"].strip()
                h = hashlib.md5(q.encode()).hexdigest()
                if h in seen or h in skip:
                    continue
                seen.add(h)
                probs.append((q, a))
        print(f"  {fp.split('/')[-1]}: {len(probs)} problems", flush=True)
        if len(probs) >= args.n_problems:
            break
    rng.shuffle(probs)
    probs = probs[: args.n_problems]
    print(f"sampling {len(probs)} problems x {args.k}", flush=True)

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            tokenize=False, add_generation_prompt=True)
        for q, _ in probs
    ]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1024, dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept = 0
    n_any = 0
    with open(args.out, "w") as f:
        for (q, gold), o in zip(probs, outs):
            cands = []
            for c in o.outputs:
                t = c.text.strip()
                m = ANSLINE.search(t)
                if not m:
                    continue
                if norm(m.group(1)) != gold:
                    continue
                if t.count("ANSWER:") != 1:
                    continue
                cands.append(t)
            if cands:
                n_any += 1
            # shortest-first: prefer concise correct chains, then dedup
            cands = sorted(set(cands), key=len)[: args.max_keep_per_problem]
            for t in cands:
                use_fs = rng.random() < args.fewshot_frac
                msgs = ([{"role": "system", "content": sysmsg}] if use_fs else []) + [
                    {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
                prompt = tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
                f.write(json.dumps({"prompt": prompt, "completion": t + "<end_of_turn>",
                                    "answer": str(gold), "source": "rft",
                                    "fewshot": use_fs}) + "\n")
                kept += 1
    print(f"solved at least once: {n_any}/{len(probs)} = {n_any/len(probs):.3f}", flush=True)
    print(f"wrote {kept} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
