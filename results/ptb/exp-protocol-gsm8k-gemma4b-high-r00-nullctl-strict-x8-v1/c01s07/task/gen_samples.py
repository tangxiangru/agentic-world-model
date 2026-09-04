#!/usr/bin/env python3
"""Rejection sampling: draw k CoTs per problem from the current policy, keep the
ones whose final ANSWER matches the reference answer."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from prep_data import MATH_PROMPT_TEMPLATE
from train_sft import BASE, build_prompt

ANS_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.M)


def norm_num(s: str):
    s = s.strip().strip("$").replace(",", "").replace("%", "").rstrip(".")
    s = s.replace("\\!", "").replace(" ", "")
    try:
        return round(float(s), 6)
    except ValueError:
        return None


def extract(text: str):
    ms = ANS_RE.findall(text)
    if not ms:
        return None
    return norm_num(ms[-1])


def load_problems(args, rng):
    probs = []
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    for rec in ds:
        probs.append((rec["question"].strip(), rec["answer"].split("####")[-1].strip(), "gsm8k"))
    if args.n_aug > 0:
        import pyarrow.parquet as pq

        files = sorted(
            glob.glob(
                "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                "snapshots/*/data/train_1M-*.parquet"
            )
        )
        pool = []
        seen = set()
        for f in files:
            df = pq.read_table(f).to_pandas()
            sub = df[df.problem_source == "augmented_gsm8k"]
            for p, a in zip(sub["problem"], sub["expected_answer"]):
                k = p.strip().lower()
                if k in seen or len(p) > 1200:
                    continue
                seen.add(k)
                pool.append((p.strip(), a.strip(), "augmented_gsm8k"))
        rng.shuffle(pool)
        probs.extend(pool[: args.n_aug])
    rng.shuffle(probs)
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft_v1.jsonl")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-aug", type=int, default=20000)
    ap.add_argument("--max-keep", type=int, default=2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probs = load_problems(args, rng)
    print(f"{len(probs)} problems, k={args.k}")

    tok = AutoTokenizer.from_pretrained(BASE)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=1536,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temp,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop_token_ids=[106, 1],
        seed=args.seed,
    )
    prompts = [
        TokensPrompt(
            prompt_token_ids=tok(
                build_prompt("", MATH_PROMPT_TEMPLATE.format(prompt=p)),
                add_special_tokens=False,
            )["input_ids"]
        )
        for p, _, _ in probs
    ]
    outs = llm.generate(prompts, sp)

    n_solved = 0
    n_kept = 0
    per_source = {}
    with open(args.out, "w") as f:
        for (problem, gold, src), o in zip(probs, outs):
            g = norm_num(gold)
            good, seen = [], set()
            for c in o.outputs:
                txt = c.text.strip()
                if c.finish_reason != "stop":
                    continue
                if extract(txt) is None or g is None or extract(txt) != g:
                    continue
                key = re.sub(r"\s+", " ", txt)
                if key in seen:
                    continue
                seen.add(key)
                good.append(txt)
            st = per_source.setdefault(src, [0, 0])
            st[1] += 1
            if good:
                n_solved += 1
                st[0] += 1
            good.sort(key=len)
            for txt in good[: args.max_keep]:
                f.write(
                    json.dumps(
                        {
                            "problem": problem,
                            "solution": txt,
                            "answer": gold,
                            "source": src,
                        }
                    )
                    + "\n"
                )
                n_kept += 1
    print(f"solved {n_solved}/{len(probs)} problems; kept {n_kept} solutions")
    for s, (a, b) in per_source.items():
        print(f"  {s}: pass@{args.k} = {a}/{b} = {a/max(b,1):.3f}")
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump({s: {"solved": a, "total": b} for s, (a, b) in per_source.items()}, f, indent=2)


if __name__ == "__main__":
    main()
