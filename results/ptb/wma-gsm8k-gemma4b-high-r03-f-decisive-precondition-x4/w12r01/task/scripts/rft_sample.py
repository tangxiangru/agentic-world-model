#!/usr/bin/env python3
"""Rejection-sampling data generation from a trained checkpoint.

Samples k completions per GSM8K *train* question with vLLM, keeps the ones whose
final ANSWER matches the gold answer, dedups, and writes rows in exactly the
format scripts/train_sft.py consumes.  No test item is ever read.
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
import fmt  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

GSM8K_TRAIN = glob.glob(
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"
)
OMI = sorted(
    glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train-*.parquet"
    )
)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def last_number(text: str) -> str | None:
    t = text.replace(",", "")
    m = _NUM.findall(t)
    return m[-1] if m else None


def same_number(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def load_questions(which, limit, seed):
    rows = []
    if which in ("gsm8k", "both"):
        for p in GSM8K_TRAIN:
            for r in pq.read_table(p).to_pylist():
                _, ans = fmt.clean_gsm8k_reasoning(r["answer"])
                rows.append({"q": r["question"].strip(), "a": ans, "src": "gsm8k_train"})
    if which in ("omi", "both"):
        seen = set()
        for p in OMI:
            for r in pq.read_table(
                p, columns=["problem", "expected_answer", "problem_source"]
            ).to_pylist():
                if r["problem_source"] != "augmented_gsm8k":
                    continue
                a = fmt.normalize_number(r["expected_answer"])
                if not re.fullmatch(r"-?\d+(\.\d+)?", a):
                    continue
                q = r["problem"].strip()
                if q in seen:
                    continue
                seen.add(q)
                rows.append({"q": q, "a": a, "src": "omi_aug"})
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", default="both", choices=["gsm8k", "omi", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--adaptive-keep", action="store_true",
                    help="keep more chains for questions the model rarely solves, fewer for the easy ones")
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qs = load_questions(args.questions, args.limit, args.seed)
    print(f"[rft] {len(qs)} questions x k={args.k}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=1536,
        dtype="bfloat16",
        enforce_eager=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    prompts = [fmt.render_prompt(r["q"]) for r in qs]
    outs = llm.generate(prompts, sp)

    kept, n_corr, n_tot = [], 0, 0
    per_q_solved = 0
    for r, o in zip(qs, outs):
        cands, seen = [], set()
        n_ok_this_q = 0
        for c in o.outputs:
            n_tot += 1
            txt = c.text.strip()
            pred = last_number(txt)
            if not same_number(pred, r["a"]):
                continue
            n_corr += 1
            n_ok_this_q += 1
            if "ANSWER:" not in txt:
                continue
            body = txt.rsplit("ANSWER:", 1)[0].strip()
            if not body or len(body) < 20:
                continue
            key = re.sub(r"\s+", " ", body)[:400]
            if key in seen:
                continue
            seen.add(key)
            cands.append(body)
        if not cands:
            continue
        per_q_solved += 1
        # Keep reasoning-diverse chains, not just the shortest: the RFT paper's gain
        # comes from distinct equation sets per question. Sort by length and take
        # evenly spaced picks so a short and a long chain both survive.
        keep_n = args.keep_per_question
        if args.adaptive_keep:
            # exp-03's conclusion: the churn is at the decision boundary, so spend
            # the data budget on questions the model only sometimes gets right.
            rate = n_ok_this_q / len(o.outputs)
            keep_n = 1 if rate >= 0.75 else (args.keep_per_question + 1 if rate <= 0.25 else args.keep_per_question)
        cands.sort(key=len)
        if len(cands) > keep_n:
            step = len(cands) / keep_n
            cands = [cands[min(len(cands) - 1, int(i * step))] for i in range(keep_n)]
        for body in cands[:keep_n]:
            kept.append(
                {
                    "prompt": fmt.render_prompt(r["q"]),
                    "target": fmt.build_target(body, r["a"]),
                    "question": r["q"],
                    "answer": r["a"],
                    "source": "rft_" + r["src"],
                    "n_shots": 0,
                }
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    print(
        f"[rft] sample-level correct {n_corr}/{n_tot} ({n_corr/max(1,n_tot):.3f}); "
        f"questions with >=1 correct {per_q_solved}/{len(qs)} ({per_q_solved/len(qs):.3f}); "
        f"kept {len(kept)} rows -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
