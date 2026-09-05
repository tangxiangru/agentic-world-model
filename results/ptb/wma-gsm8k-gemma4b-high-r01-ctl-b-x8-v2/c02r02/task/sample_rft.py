#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per training question
from a checkpoint, keep the ones whose final answer matches gold.

Questions come from GSM8K's *train* split and from OpenMathInstruct-2's
gsm8k/augmented_gsm8k problems (which are themselves seeded from GSM8K train).
Nothing here touches the test split.

The correctness test is the grader's own: inspect_ai.scorer._common.match_str
with numeric=True and location="end", i.e. the last numeric token of the
completion.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

import fmt

OMI_GLOB = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data/train-*.parquet"
)
NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def grader_match(completion: str, gold: str) -> bool:
    from inspect_ai.scorer._common import match_str

    _, ok = match_str(value=completion, target=gold, location="end",
                      ignore_case=True, numeric=True)
    return ok


def clean_number(a: str):
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    return a[:-2] if a.endswith(".0") else a


def question_pool(n_omi: int, seed: int):
    from datasets import load_dataset

    out = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        a = clean_number(r["answer"].rpartition("####")[2])
        if a:
            out.append((r["question"], a))
    n_gsm8k = len(out)

    seen = set(q for q, _ in out)
    omi = []
    for f in sorted(glob.glob(OMI_GLOB)):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=50_000,
                                 columns=["problem", "expected_answer", "problem_source"]):
            d = b.to_pydict()
            for p, a, s in zip(d["problem"], d["expected_answer"], d["problem_source"]):
                if s not in ("gsm8k", "augmented_gsm8k") or p in seen:
                    continue
                seen.add(p)
                a2 = clean_number(a)
                if a2:
                    omi.append((p, a2))
    random.Random(seed).shuffle(omi)
    out += omi[:n_omi]
    print(f"question pool: {n_gsm8k} gsm8k-train + {min(n_omi, len(omi))} OMI "
          f"(of {len(omi)} unique available) = {len(out)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi", type=int, default=40_000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pool = question_pool(args.n_omi, args.seed)
    if args.limit:
        pool = pool[: args.limit]
    prompts = [fmt.render_prompt(q) for q, _ in pool]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=1024, enforce_eager=False, seed=args.seed,
              max_num_seqs=args.max_num_seqs, disable_log_stats=True)
    # no per-request seed: it forces vLLM onto a per-sequence RNG path and
    # roughly halved throughput in the first attempt.
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept = defaultdict(list)
    n_corr = n_tot = 0
    solved = 0
    stats_path = args.out.replace(".jsonl", "_stats.json")
    per_q = []
    for (q, gold), o in zip(pool, outs):
        texts = [c.text.strip() for c in o.outputs]
        good = []
        for t in texts:
            n_tot += 1
            if not t or fmt.ANSWER_MARKER not in t:
                continue
            if t.count("ANSWER:") != 1:
                continue
            if grader_match(t, gold):
                n_corr += 1
                good.append(t)
        per_q.append(len(good))
        if good:
            solved += 1
        seen = set()
        uniq = []
        for t in good:
            key = re.sub(r"\s+", " ", t)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        uniq.sort(key=len)  # prefer the shortest correct derivations
        # frontier weighting: a question the model already solves every time
        # teaches almost nothing, so keep one solution for it and two for the
        # ones it only sometimes gets right.
        n_keep = 1 if len(good) > args.k // 2 else args.keep_per_question
        kept[q] = (uniq[:n_keep], gold)

    rows = []
    for q, (sols, gold) in kept.items():
        for s in sols:
            rows.append((q, s))
    random.Random(args.seed).shuffle(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for q, s in rows:
            p, c = fmt.render_example(q, s)
            f.write(json.dumps({"prompt": p, "completion": c, "fewshot": False}) + "\n")
    ck = args.out.replace(".jsonl", "_for_contamcheck.jsonl")
    with open(ck, "w") as f:
        for q, s in rows:
            f.write(json.dumps({"question": q, "answer": s}) + "\n")

    stats = {
        "questions": len(pool),
        "samples": n_tot,
        "correct_samples": n_corr,
        "sample_accuracy": n_corr / max(n_tot, 1),
        "questions_with_at_least_one_correct": solved,
        "pass_at_k": solved / max(len(pool), 1),
        "rows_written": len(rows),
        "model": args.model,
        "k": args.k,
        "temperature": args.temperature,
    }
    json.dump(stats, open(stats_path, "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    print("wrote", args.out, ck, stats_path, flush=True)


if __name__ == "__main__":
    main()
