#!/usr/bin/env python3
"""Rejection sampling from a fine-tuned checkpoint on the gsm8k TRAIN split.

Questions come from openai/gsm8k main/train only - never the test split. The
model's own samples are kept when the grader's own scorer reads the gold answer
out of them, so the resulting file is on-policy and verified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import END, GSM8K_SNAPSHOT, render_prompt, render_target  # noqa: E402

from inspect_ai.scorer._common import match_str  # noqa: E402

TRAIN = os.path.join(GSM8K_SNAPSHOT, "main", "train-00000-of-00001.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = pq.read_table(TRAIN).to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    qs, golds = [], []
    for r in rows:
        q = r["question"].strip()
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        qs.append(q)
        golds.append(gold)
    print(f"train questions: {len(qs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = [
        tok(render_prompt(q), add_special_tokens=False)["input_ids"] for q in qs
    ]
    print(f"prompt tokens p50={sorted(len(p) for p in prompt_ids)[len(qs)//2]}", flush=True)

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sp)

    kept = 0
    n_correct_any = 0
    stats = {"gen": 0, "correct": 0, "dupe": 0, "bad_format": 0}
    with open(args.out, "w") as fh, open(args.out + ".questions.jsonl", "w") as qh:
        for q, gold, o in zip(qs, golds, outs):
            seen, taken, any_ok = set(), 0, False
            for c in o.outputs:
                stats["gen"] += 1
                body = c.text.strip()
                if not body or "ANSWER:" not in body:
                    stats["bad_format"] += 1
                    continue
                if body.count("ANSWER:") != 1:
                    stats["bad_format"] += 1
                    continue
                _, ok = match_str(value=body, target=gold, location="end", numeric=True)
                if not ok:
                    continue
                stats["correct"] += 1
                any_ok = True
                if taken >= args.keep_per_question:
                    continue
                key = hash(body)
                if key in seen:
                    stats["dupe"] += 1
                    continue
                seen.add(key)
                taken += 1
                kept += 1
                fh.write(
                    json.dumps(
                        {
                            "prompt": render_prompt(q),
                            "completion": render_target(body),
                            "answer": gold,
                            "source": "rft",
                            "fewshot": False,
                        }
                    )
                    + "\n"
                )
                qh.write(json.dumps({"text": q + "\n" + body}) + "\n")
            n_correct_any += int(any_ok)

    stats["kept"] = kept
    stats["questions_with_a_correct_sample"] = n_correct_any
    stats["questions"] = len(qs)
    stats["pass_at_%d" % args.n_samples] = round(n_correct_any / max(1, len(qs)), 4)
    stats["sample_accuracy"] = round(stats["correct"] / max(1, stats["gen"]), 4)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
