#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data: sample solutions from a checkpoint on
GSM8K *train* problems, keep the ones whose final answer is right.

Nothing here touches the benchmark test split. Problems come from
openai/gsm8k main/train minus the 300-item local holdout.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from eval_local import MATH_PROMPT_TEMPLATE, graded_answer, norm_num

END_OF_TURN = "<end_of_turn>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()

    holdout = {json.loads(l)["question"].strip().lower()
               for l in open("data/dev_train_holdout.jsonl")}
    ds = load_dataset("openai/gsm8k", "main")["train"]
    items = []
    for r in ds:
        if r["question"].strip().lower() in holdout:
            continue
        gold = r["answer"].rpartition("####")[2].strip().replace(",", "")
        items.append({"question": r["question"].strip(), "gold": gold})
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} train problems")

    prompts = [tok.apply_chat_template(
        [{"role": "user",
          "content": MATH_PROMPT_TEMPLATE.format(prompt=it["question"])}],
        tokenize=False, add_generation_prompt=True) for it in items]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=2048, dtype="bfloat16")
    sp = SamplingParams(temperature=args.temp, top_p=0.95, max_tokens=args.max_tokens,
                        n=args.k, stop_token_ids=[106, 1], seed=0)
    outs = llm.generate(prompts, sp)

    rng = random.Random(0)
    rows, n_solved = [], 0
    for it, o in zip(items, outs):
        gold = norm_num(it["gold"])
        good = []
        for c in o.outputs:
            if c.finish_reason == "length":
                continue
            text = c.text.strip()
            if graded_answer(text) != gold:
                continue
            if text.count("ANSWER:") != 1:
                continue
            if not re.search(r"\nANSWER: [^\n]+$", text):
                continue
            good.append(text)
        if not good:
            continue
        n_solved += 1
        # prefer distinct reasoning: dedup on the normalised body
        seen, uniq = set(), []
        for g in sorted(good, key=len):
            key = re.sub(r"\s+", " ", g)[:400]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(g)
        rng.shuffle(uniq)
        for g in uniq[: args.keep_per_problem]:
            rows.append({"question": it["question"], "target": g + END_OF_TURN,
                         "answer": it["gold"], "src": "rft"})

    print(f"solved {n_solved}/{len(items)} ({n_solved/len(items):.3f}); "
          f"kept {len(rows)} rows")
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
