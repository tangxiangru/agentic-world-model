#!/usr/bin/env python3
"""Sample solutions from a trained checkpoint and keep the ones that reach the gold answer.

Questions come from gsm8k TRAIN and from OpenMathInstruct-2's augmented_gsm8k
problems (both train-derived, never the test split). Prompts are rendered with
build_data.render_prompt, i.e. byte-identical to the grader's prompt, and fed to
vLLM as token ids so the <bos> is not doubled.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

from build_data import EOT, INT_RE, norm_q, render_prompt

SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    hits = NUM.findall(text.replace(",", ""))
    if not hits:
        return None
    v = hits[-1].rstrip(".")
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return str(int(f)) if f == int(f) else v
    except (ValueError, OverflowError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm8k", type=int, default=7473)
    ap.add_argument("--n-augmented", type=int, default=9000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)

    items: list[dict] = []
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for s in list(gsm)[: args.n_gsm8k]:
        items.append(
            {
                "q": s["question"],
                "a": s["answer"].split("####")[-1].strip().replace(",", ""),
                "src": "gsm8k_train",
            }
        )
    if args.n_augmented:
        seen = {norm_q(i["q"]) for i in items}
        pool = []
        for f in sorted(
            glob.glob(
                "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                "snapshots/*/data/train_1M-*.parquet"
            )
        ):
            for r in pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pylist():
                if r["problem_source"] != "augmented_gsm8k":
                    continue
                ans = (r["expected_answer"] or "").strip().replace(",", "")
                if not INT_RE.match(ans):
                    continue
                k = norm_q(r["problem"])
                if k in seen:
                    continue
                seen.add(k)
                pool.append({"q": r["problem"], "a": ans, "src": "augmented_gsm8k"})
        rng.shuffle(pool)
        items.extend(pool[: args.n_augmented])

    print(f"{len(items)} questions to sample, {args.samples} each")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=2560,
        dtype="bfloat16",
        enforce_eager=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    prompts = [
        {"prompt_token_ids": tok(render_prompt(i["q"], None), add_special_tokens=False)["input_ids"]}
        for i in items
    ]
    outs = llm.generate(prompts, sp)
    # dump raw generations first: post-processing must never be able to lose GPU work
    with open(args.out + ".raw.jsonl", "w") as fh:
        for item, out in zip(items, outs):
            fh.write(json.dumps({"q": item["q"], "a": item["a"], "src": item["src"],
                                 "gens": [c.text for c in out.outputs]}) + "\n")
    print("raw generations written to", args.out + ".raw.jsonl", flush=True)

    kept: list[dict] = []
    per_q: defaultdict = defaultdict(int)
    n_correct = n_total = 0
    solved = 0
    for item, out in zip(items, outs):
        any_ok = False
        texts = []
        for c in out.outputs:
            n_total += 1
            t = c.text.strip()
            if last_number(t) != item["a"]:
                continue
            if "ANSWER:" not in t or t.count("ANSWER:") != 1:
                continue
            n_correct += 1
            any_ok = True
            texts.append(t)
        solved += int(any_ok)
        # prefer the shortest correct solutions: less rambling, fewer stray numbers
        texts = sorted(set(texts), key=len)[: args.keep_per_question]
        for t in texts:
            kept.append(
                {
                    "prompt": render_prompt(item["q"], None),
                    "completion": t + EOT,
                    "src": "rft:" + item["src"],
                    "answer": item["a"],
                }
            )
    print(
        f"pass@1 {n_correct/max(n_total,1):.3f}  solved-at-least-once "
        f"{solved}/{len(items)} = {solved/len(items):.3f}  kept {len(kept)} rows"
    )
    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
