#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per problem from a trained checkpoint,
keep the ones whose final number matches the gold answer.

Problems come from two places, both GSM8K *train*-derived and both filtered
against data/dev250.jsonl:
  * the gsm8k train split,
  * the augmented_gsm8k problems already in data/sft_v2.jsonl.

The prompt is rendered exactly as in training (zero-shot, templates/gemma3.jinja)
so the kept completions are in-distribution for the next SFT pass. Correctness
uses inspect's rule: the last number in the completion, normalised.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

from eval_probe import PROMPT_TEMPLATE, last_number, norm_num, strip_numeric_punctuation

END_OF_TURN = "<end_of_turn>"
INT_RE = re.compile(r"^-?\d+$")


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def load_problems(args, dev: set[str]) -> list[dict]:
    """Problems to sample on: the GSM8K train split plus the augmented_gsm8k
    problems already in data/sft_v2.jsonl.

    Sampling fresh problems from later OpenMathInstruct-2 shards was tried and
    abandoned: shards 6-11 yielded only 749 problems not already present in
    shards 0-5 (data/sft_v3.jsonl), i.e. the dataset's ~70k unique
    gsm8k-derived problems are already all in sft_v2. RFT is therefore run on
    the same problems, which is what the RFT/STaR recipe does anyway.
    """
    probs: list[dict] = []
    seen: set[str] = set()
    from datasets import load_dataset
    train = load_dataset("openai/gsm8k", "main")["train"]
    for r in train:
        nq = norm_q(r["question"])
        if nq in dev or nq in seen:
            continue
        seen.add(nq)
        probs.append({"question": r["question"].strip(),
                      "gold": r["answer"].split("####")[-1].strip(),
                      "src": "gsm8k_train"})
    print(f"gsm8k train problems: {len(probs)}")

    aug = []
    for line in open("data/sft_v2.jsonl"):
        r = json.loads(line)
        q = r["prompt"].split("\n\nRemember to put your answer")[0]
        q = q.split("is the answer to the problem.\n\n", 1)[-1].strip()
        nq = norm_q(q)
        if nq in dev or nq in seen:
            continue
        seen.add(nq)
        aug.append({"question": q, "gold": r["answer"], "src": "sft_v2_problem"})
    random.Random(args.seed).shuffle(aug)
    probs.extend(aug)
    print(f"augmented problems available: {len(aug)}")

    if args.uncovered:
        # round 2: only problems the previous round left without a verified
        # solution -- either never sampled, or sampled and never solved. Those
        # are where a stronger policy can add something the mixture lacks.
        covered = set()
        for line in open(args.uncovered):
            r = json.loads(line)
            q = r["prompt"].split("\n\nRemember to put your answer")[0]
            q = q.split("is the answer to the problem.\n\n", 1)[-1].strip()
            covered.add(norm_q(q))
        before = len(probs)
        probs = [p for p in probs if norm_q(p["question"]) not in covered]
        print(f"uncovered filter: {before} -> {len(probs)} "
              f"(dropped {before - len(probs)} already solved in {args.uncovered})")

    random.Random(args.seed).shuffle(probs)
    return probs[: args.n_problems] if args.n_problems > 0 else probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-problems", type=int, default=-1)
    ap.add_argument("--uncovered", default=None,
                    help="jsonl of already-solved rows; sample only problems absent from it")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = {norm_q(json.loads(l)["question"]) for l in open("data/dev250.jsonl")}
    probs = load_problems(args, dev)
    random.Random(args.seed).shuffle(probs)
    print(f"total problems to sample: {len(probs)}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tpl = open("templates/gemma3.jinja").read()
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=p["question"])}],
            chat_template=tpl, tokenize=False, add_generation_prompt=True)
        for p in probs
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=2048)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        top_k=64, max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)

    rows, n_correct, n_tot = [], 0, 0
    for p, o in zip(probs, outs):
        gold = norm_num(strip_numeric_punctuation(p["gold"].casefold()))
        kept = []
        bodies = set()
        for c in o.outputs:
            n_tot += 1
            if c.finish_reason != "stop":
                continue
            text = c.text.strip()
            pred = last_number(text)
            if pred is None or norm_num(pred) != gold:
                continue
            n_correct += 1
            # exactly one answer marker, and the last line is that marker
            if text.lower().count("answer:") != 1:
                continue
            if not re.search(r"ANSWER:\s*-?\d+\s*$", text):
                continue
            key = re.sub(r"\s+", " ", text)[:150]
            if key in bodies:
                continue
            bodies.add(key)
            kept.append(text)
            if len(kept) >= args.max_per_problem:
                break
        for text in kept:
            rows.append({
                "prompt": PROMPT_TEMPLATE.format(prompt=p["question"]),
                "completion": text + END_OF_TURN,
                "answer": p["gold"],
                "source": "rft:" + p["src"],
                "system": None,
            })

    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({
        "problems": len(probs), "samples": n_tot,
        "sample_accuracy": n_correct / max(1, n_tot),
        "kept_rows": len(rows),
        "problems_with_at_least_one": len({r["prompt"] for r in rows}),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
