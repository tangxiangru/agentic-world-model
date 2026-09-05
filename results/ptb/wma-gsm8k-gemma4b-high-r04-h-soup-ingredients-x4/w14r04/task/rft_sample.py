#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data from the model's own solutions.

Samples k completions per GSM8K-TRAIN question (never the held-out probe, never
the benchmark test split), keeps only those whose graded last number equals the
gold answer, dedups, and writes rows in the same schema build_data.py emits.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from transformers import AutoTokenizer

from probe_eval import MATH_PROMPT_TEMPLATE, last_number


def norm(s: str) -> str:
    """Cheap structural key for dedup: numbers kept, prose collapsed."""
    return re.sub(r"[^0-9=+\-*/.]", "", s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", default="/home/ben/task/data/gsm8k_train_rest.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-q", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--fewshot", default="/home/ben/task/data/fewshot_system.txt")
    ap.add_argument("--gsm-rest", default="/home/ben/task/data/gsm8k_train_rest.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("/home/ben/task/templates/gemma3.jinja").read()

    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept = defaultdict(list)
    n_correct = 0
    n_total = 0
    solved = 0
    for r, o in zip(rows, outs):
        gold = format(float(r["gold"].replace(",", "")), ".5g")
        any_ok = False
        seen = set()
        for c in o.outputs:
            n_total += 1
            text = c.text.strip()
            if c.finish_reason == "length":
                continue
            if text.count("ANSWER:") != 1 or "\\boxed" in text or "####" in text:
                continue
            if last_number(text) != gold:
                continue
            n_correct += 1
            any_ok = True
            key = norm(text)
            if key in seen:
                continue
            seen.add(key)
            if len(kept[r["id"]]) < args.keep_per_q:
                kept[r["id"]].append((r["question"], text))
        solved += any_ok

    # exemplar pool for the 10-shot prefix minority, same construction as build_data
    exemplars = []
    for r in (json.loads(l) for l in open(args.gsm_rest)):
        exemplars.append((r["question"], r["answer"].split("####")[0].strip(), r["gold"]))

    out_rows = []
    for _id, items in kept.items():
        for q, sol in items:
            if rng.random() < args.fewshot_frac:
                shots = rng.sample(exemplars, 10)
                system = "\n\n".join(
                    f"{a}\n\nReasoning:\n{b}\n\nANSWER: {c}" for a, b, c in shots
                )
                nshot = 10
            else:
                system, nshot = None, 0
            out_rows.append({
                "system": system,
                "user": MATH_PROMPT_TEMPLATE.format(prompt=q.strip()),
                "target": sol + "<end_of_turn>",
                "src": "rft_self",
                "nshot": nshot,
            })
    rng.shuffle(out_rows)
    with open(args.out, "w") as f:
        for o in out_rows:
            f.write(json.dumps(o) + "\n")
    print(f"questions {len(rows)}  samples {n_total}  correct {n_correct} "
          f"({n_correct/max(n_total,1):.3f})  solved-at-least-once {solved} "
          f"({solved/len(rows):.3f})  rows written {len(out_rows)} -> {args.out}")


if __name__ == "__main__":
    main()
