#!/usr/bin/env python3
"""Rejection-sampling round: sample k solutions per problem from a checkpoint,
keep the ones whose graded answer is correct, write a new SFT jsonl.

Prompts are rendered with the grader's own templates/gemma3.jinja and the
grader's MATH_PROMPT_TEMPLATE; completions are graded with a byte-copy of
inspect's match(numeric=True, location=end).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import END_OF_TURN, grade, load_template, user_prompt  # noqa: E402

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def load_problems(n_omi: int, seed: int):
    from datasets import load_dataset

    probs = []
    gsm = load_dataset("openai/gsm8k", "main")["train"]
    for rec in gsm:
        if "####" not in rec["answer"]:
            continue
        ans = rec["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        if NUM_RE.match(ans):
            probs.append({"question": rec["question"], "answer": ans, "src": "gsm8k_train"})
    n_gsm = len(probs)

    if n_omi > 0:
        import pyarrow.parquet as pq

        seen = set()
        pool = []
        for path in sorted(glob.glob(OMI_GLOB)):
            t = pq.read_table(path, columns=["problem", "expected_answer", "problem_source"])
            d = t.to_pydict()
            for prob, ans, src in zip(d["problem"], d["expected_answer"], d["problem_source"]):
                if src != "augmented_gsm8k":
                    continue
                ans = (ans or "").strip().replace(",", "")
                if not NUM_RE.match(ans) or prob in seen or len(prob) > 2000:
                    continue
                seen.add(prob)
                pool.append({"question": prob, "answer": ans, "src": "omi_aug"})
        random.Random(seed).shuffle(pool)
        probs.extend(pool[:n_omi])
    print(f"problems: {len(probs)} ({n_gsm} gsm8k train + {len(probs)-n_gsm} omi augmented)", flush=True)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-omi", type=int, default=15000)
    ap.add_argument("--max-keep", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    template = load_template()
    probs = load_problems(args.n_omi, args.seed)

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": user_prompt(p["question"])}],
            chat_template=template, tokenize=False, add_generation_prompt=True,
        )
        for p in probs
    ]

    llm = LLM(
        model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048, seed=args.seed, enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k, temperature=args.temperature, top_p=0.95, top_k=64,
        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    n_solved = 0
    n_rows = 0
    per_src = {}
    with open(args.out, "w") as f:
        for p, o in zip(probs, outs):
            cands = []
            for c in o.outputs:
                txt = c.text.strip()
                if not txt or not grade(txt, p["answer"]):
                    continue
                if "ANSWER:" not in txt:
                    continue
                cands.append(txt)
            if not cands:
                continue
            n_solved += 1
            uniq = list(dict.fromkeys(cands))
            rng.shuffle(uniq)
            for txt in uniq[: args.max_keep]:
                f.write(json.dumps({
                    "messages": [{"role": "user", "content": user_prompt(p["question"])}],
                    "completion": txt + END_OF_TURN,
                    "answer": p["answer"],
                    "src": "rft:" + p["src"],
                }) + "\n")
                n_rows += 1
                per_src[p["src"]] = per_src.get(p["src"], 0) + 1
    stats = {
        "n_problems": len(probs), "k": args.k,
        "n_problems_with_a_correct_sample": n_solved,
        "solve_rate_at_k": n_solved / max(1, len(probs)),
        "n_rows_written": n_rows, "rows_per_source": per_src,
    }
    print(json.dumps(stats, indent=2))
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
