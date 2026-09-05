#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per training question, keep the ones
whose final 'ANSWER: N' matches gold, write them as SFT rows.

Questions come from the GSM8K TRAIN split and from OpenMathInstruct-2
gsm8k-derived problems (also train-derived). Never from the test split.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)")


def norm(x: str) -> str | None:
    x = str(x).strip().replace(",", "").replace("$", "")
    if len(x) > 20:          # a 20+ digit "answer" is degenerate output, not a number
        return None
    try:
        v = float(x)
    except (ValueError, OverflowError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    try:
        return str(int(v)) if v == int(v) else str(v)
    except (OverflowError, ValueError):
        return None


def extract(text: str) -> str | None:
    m = ANS.findall(text)
    return norm(m[-1]) if m else None


def load_questions(args) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    if args.gsm_train:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="train")
        for r in ds:
            a = norm(r["answer"].split("####")[-1])
            if a is None:
                continue
            q = r["question"].strip()
            if q in seen:
                continue
            seen.add(q)
            out.append({"question": q, "answer": a, "source": "gsm8k-train"})
    if args.pool:
        for line in open(args.pool):
            r = json.loads(line)
            q = r["question"].strip()
            if q in seen:
                continue
            seen.add(q)
            out.append({"question": q, "answer": r["answer"], "source": "omi2-pool"})
    rng = random.Random(args.seed)
    rng.shuffle(out)
    return out[: args.n_questions] if args.n_questions > 0 else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default=None, help="jsonl of extra {question,answer}")
    ap.add_argument("--gsm-train", type=int, default=1)
    ap.add_argument("--n-questions", type=int, default=16000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    qs = load_questions(args)
    print("questions", len(qs))

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=1536,
        dtype="bfloat16",
        seed=args.seed,
        generation_config="vllm",
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    prompts = [fmt.render_prompt(q["question"], None) for q in qs]
    outs = llm.generate(prompts, sp)

    # persist the raw draws before any filtering: a crash in the filter must not
    # cost the generation pass again (it cost 0.6 h once)
    raw_path = args.out + ".raw.jsonl"
    with open(raw_path, "w") as f:
        for q, o in zip(qs, outs):
            f.write(json.dumps({"q": q, "texts": [c.text for c in o.outputs]}) + "\n")
    print("raw ->", raw_path, flush=True)

    kept: list[dict] = []
    solved = 0
    per_q_correct = []
    rng = random.Random(args.seed)
    for q, o in zip(qs, outs):
        cands = []
        for c in o.outputs:
            t = c.text.strip()
            if extract(t) == q["answer"]:
                cands.append(t)
        per_q_correct.append(len(cands))
        if not cands:
            continue
        solved += 1
        # dedup, prefer concise but not degenerate
        uniq = list(dict.fromkeys(cands))
        uniq.sort(key=len)
        pick = uniq[: args.keep_per_question]
        for p in pick:
            kept.append(
                {
                    "question": q["question"],
                    "completion": p + fmt.EOT,
                    "answer": q["answer"],
                    "source": "rft:" + q["source"],
                }
            )
    rng.shuffle(kept)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {
        "n_questions": len(qs),
        "k": args.k,
        "solved_at_least_once": solved,
        "pass_at_k": solved / max(1, len(qs)),
        "mean_correct_per_question": sum(per_q_correct) / max(1, len(per_q_correct)),
        "kept_rows": len(kept),
        "out": args.out,
    }
    print(json.dumps(stats, indent=2))
    if args.stats:
        json.dump(stats, open(args.stats, "w"), indent=2)


if __name__ == "__main__":
    main()
