#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per training question from a
checkpoint, keep the ones whose final number equals the gold answer.

Questions and gold answers come from the SFT pool (OpenMathInstruct-2 gsm8k /
augmented_gsm8k problems, all from training splits). No benchmark item is used.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

EOT = "<end_of_turn>"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    m = NUM.findall(text.replace(",", ""))
    if not m:
        return None
    v = m[-1].rstrip(".")
    if v.endswith(".0"):
        v = v[:-2]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True, help="jsonl with prompt/question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-questions", type=int, default=40000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep", type=int, default=2, help="max kept solutions per question")
    ap.add_argument("--keep-easy-prob", type=float, default=0.34,
                    help="probability of keeping one solution for a question solved k/k times")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    seen: dict[str, dict] = {}
    with open(args.pool) as f:
        for line in f:
            r = json.loads(line)
            if r.get("n_shots"):
                continue  # use the plain zero-shot prompt for sampling
            if r["question"] not in seen:
                seen[r["question"]] = r
    items = list(seen.values())
    rng.shuffle(items)
    items = items[: args.n_questions]
    print(f"{len(items)} questions, k={args.k}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    # tokenize explicitly: vLLM would add a second BOS if handed raw strings
    prompts = [
        {"prompt_token_ids": [tok.bos_token_id]
         + tok(it["prompt"], add_special_tokens=False)["input_ids"]}
        for it in items
    ]
    outs = llm.generate(prompts, sp)

    n_correct = n_total = 0
    per_rate: dict[int, int] = defaultdict(int)
    kept = []
    for it, out in zip(items, outs):
        gold = it["answer"]
        good = []
        for c in out.outputs:
            text = c.text.strip()
            n_total += 1
            if last_number(text) == gold and "ANSWER:" in text and text.count("ANSWER:") == 1:
                n_correct += 1
                good.append(text)
        per_rate[len(good)] += 1
        if not good:
            continue
        # dedup identical samples, prefer shorter (less rambling) solutions
        good = sorted(set(good), key=len)
        if len(good) == args.k and rng.random() > args.keep_easy_prob:
            continue  # question the model already always solves
        for text in good[: args.keep]:
            kept.append(
                {
                    "prompt": it["prompt"],
                    "completion": text + EOT,
                    "question": it["question"],
                    "answer": gold,
                    "src": "rft",
                }
            )

    print(f"sample accuracy {n_correct}/{n_total} = {n_correct/max(1,n_total):.3f}")
    print("questions by number of correct samples:", dict(sorted(per_rate.items())))
    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(kept), "rows to", args.out)
    with open(args.out.replace(".jsonl", "") + ".contam.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps({"text": r["question"] + "\n" + r["completion"][: -len(EOT)]}) + "\n")


if __name__ == "__main__":
    main()
