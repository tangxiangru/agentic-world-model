#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per problem from a checkpoint,
keep the ones whose 'ANSWER: N' line matches gold.

Problems come from the GSM8K *train*-derived pool (OpenMathInstruct-2 gsm8k /
augmented_gsm8k), never from the test split. Prompts are rendered with the same
scripts/render_sft.render the trainer uses, so what is sampled is exactly what
the grader will ask for.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import MATH_PROMPT_TEMPLATE, STOP_TOKEN, norm_num  # noqa: E402
from render_sft import render  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def answer_of(text: str) -> str | None:
    m = re.findall(r"ANSWER:\s*([^\n]*)", text)
    if not m:
        return None
    tok = m[-1].strip().split()
    return norm_num(tok[0]) if tok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-problems", type=int, default=24000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--only-hard", action="store_true",
                    help="keep only problems where at least one but not all samples are correct")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/rft.jsonl"))
    ap.add_argument("--stats-out", default=os.path.join(ROOT, "analysis/rft_stats.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    shards = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/*.parquet"
        )
    )
    seen: set[str] = set()
    pool: list[tuple[str, str]] = []
    for sh in shards:
        df = pq.read_table(sh).to_pandas()
        df = df[df["problem_source"].isin(["gsm8k", "augmented_gsm8k"])]
        for problem, exp in zip(df["problem"], df["expected_answer"]):
            p = problem.strip()
            if p in seen:
                continue
            g = norm_num(exp)
            if g is None:
                continue
            seen.add(p)
            pool.append((p, g))
        if len(pool) >= args.n_problems * 3:
            break
    rng.shuffle(pool)
    pool = pool[: args.n_problems]
    print(f"{len(pool)} problems", flush=True)

    from vllm import LLM, SamplingParams

    # render() already emits the template's <bos>; vLLM's generate() would add a
    # second one if handed a string, so tokenize here with add_special_tokens=False
    # and pass token ids.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [render(MATH_PROMPT_TEMPLATE.replace("{prompt}", p), "")[0] for p, _ in pool]
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    assert enc[0][0] == tok.bos_token_id and enc[0][1] != tok.bos_token_id, "bos handling"
    prompts = [{"prompt_token_ids": ids} for ids in enc]
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=None,
    )
    outs = llm.generate(prompts, sp)

    stats = defaultdict(int)
    n_correct_hist = defaultdict(int)
    kept = 0
    with open(args.out, "w") as f:
        for (problem, gold), o in zip(pool, outs):
            texts = [c.text for c in o.outputs]
            good = [t for t in texts if answer_of(t) == gold]
            n_correct_hist[len(good)] += 1
            stats["samples"] += len(texts)
            stats["correct"] += len(good)
            if not good:
                stats["problem_all_wrong"] += 1
                continue
            if args.only_hard and len(good) == len(texts):
                stats["problem_all_right_skipped"] += 1
                continue
            uniq, sigs = [], set()
            for t in sorted(good, key=len):
                sig = re.sub(r"\s+", " ", t).strip()
                if sig in sigs:
                    continue
                sigs.add(sig)
                uniq.append(t)
            for t in uniq[: args.keep_per_problem]:
                body = t.strip()
                if body.endswith(STOP_TOKEN):
                    body = body[: -len(STOP_TOKEN)].strip()
                f.write(
                    json.dumps(
                        {
                            "prompt": MATH_PROMPT_TEMPLATE.replace("{prompt}", problem),
                            "completion": body + STOP_TOKEN,
                            "answer": gold,
                            "source": "synthetic:self",
                            "n_shots": 0,
                        }
                    )
                    + "\n"
                )
                kept += 1
    stats["kept"] = kept
    stats["pass_rate"] = stats["correct"] / max(1, stats["samples"])
    stats["n_correct_hist"] = dict(sorted(n_correct_hist.items()))
    json.dump(stats, open(args.stats_out, "w"), indent=2, default=str)
    print(json.dumps(stats, indent=2, default=str))
    print(f"wrote {kept} rows to {args.out}")


if __name__ == "__main__":
    main()
