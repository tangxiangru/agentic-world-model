#!/usr/bin/env python3
"""Rejection-sampling data: sample K solutions per GSM8K *train* problem from a
trained checkpoint, keep the ones whose final number equals the gold answer.

Only the GSM8K train split is read. Output rows have the same schema as
data/sft_v1.jsonl so scripts/train_sft.py can consume them unchanged.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_format as EF  # noqa: E402

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"
STOP = "<end_of_turn>"
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> str | None:
    ms = NUM.findall(text)
    if not ms:
        return None
    v = ms[-1].replace(",", "")
    if v.endswith(".0"):
        v = v[:-2]
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gt = pd.read_parquet(sorted(glob.glob(GSM8K_TRAIN))[0])
    if args.limit:
        gt = gt.iloc[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)

    prompts, golds, questions = [], [], []
    for _, r in gt.iterrows():
        q = r["question"].strip()
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        prompts.append(EF.render_prompt(tok, q, None))
        golds.append(gold)
        questions.append(q)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1024, dtype="bfloat16", seed=args.seed,
              enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop=[STOP], seed=args.seed)
    outs = llm.generate(prompts, sp)

    rows, n_solved, n_kept = [], 0, 0
    stats = {"total": len(prompts), "any_correct": 0}
    for q, gold, o in zip(questions, golds, outs):
        kept = 0
        seen = set()
        for c in o.outputs:
            txt = c.text.strip()
            if not txt:
                continue
            got = last_number(txt)
            if got is None or got != gold:
                continue
            key = txt[:160]
            if key in seen:
                continue
            seen.add(key)
            if kept >= args.keep_per_problem:
                continue
            rows.append({"question": q, "target": txt + STOP, "answer": gold,
                         "source": "rft_self"})
            kept += 1
            n_kept += 1
        if kept:
            n_solved += 1
    stats["any_correct"] = n_solved
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    doc_out = args.out.replace(".jsonl", "_docs.jsonl")
    with open(doc_out, "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
    print(json.dumps({**stats, "kept_rows": n_kept,
                      "solve_rate": n_solved / max(1, len(prompts))}, indent=2))
    print("wrote", args.out, doc_out)


if __name__ == "__main__":
    main()
