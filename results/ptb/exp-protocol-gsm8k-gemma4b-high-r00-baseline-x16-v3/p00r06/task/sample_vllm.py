#!/usr/bin/env python3
"""Offline vLLM sampling against the grader's own prompt + chat template.

Two uses:
  --mode probe   score a held-out jsonl ({id,question,gold}) zero-shot or k-shot
  --mode rft     sample k completions per question and keep the correct ones
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS, SOT, EOT = "<bos>", "<start_of_turn>", "<end_of_turn>"


def fewshot_prefix(n: int = 10, seed: int = 42) -> str:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed).select(range(n))
    parts = []
    for r in ds:
        body, final = r["answer"].rsplit("####", 1)
        body = re.sub(r"<<[^>]*>>", "", body).strip()
        parts.append(f"{r['question']}\n\nReasoning:\n{body}\n\nANSWER: {final.strip()}")
    return "\n\n".join(parts)


def build_prompt(question: str, system: str | None) -> str:
    first = f"{system.strip()}\n\n" if system else ""
    user = MATH_PROMPT_TEMPLATE.format(prompt=question.strip())
    return f"{BOS}{SOT}user\n{first}{user}{EOT}\n{SOT}model\n"


def last_number(text: str) -> str | None:
    """What match(location='end', numeric=True) reads: the last number in the output."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["probe", "rft"], default="probe")
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.input)]
    if args.limit:
        rows = rows[: args.limit]
    sysmsg = fewshot_prefix(args.fewshot) if args.fewshot else None
    prompts = [build_prompt(r["question"], sysmsg) for r in rows]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=0,
    )
    outs = llm.generate(prompts, sp)

    if args.mode == "probe":
        ok = 0
        recs = []
        for r, o in zip(rows, outs):
            t = o.outputs[0].text
            c = same(last_number(t), r["gold"])
            ok += c
            recs.append({"id": r["id"], "gold": r["gold"], "correct": bool(c), "text": t})
        acc = ok / len(rows)
        print(json.dumps({"model": args.model, "n": len(rows), "fewshot": args.fewshot,
                          "temperature": args.temperature, "accuracy": acc}, indent=2))
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "n": len(rows), "fewshot": args.fewshot,
                       "temperature": args.temperature, "accuracy": acc, "samples": recs}, f)
        return

    # rft: keep correct, distinct completions, capped per question
    kept = 0
    rng = random.Random(0)
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            cands = []
            seen = set()
            for c in o.outputs:
                t = c.text.strip()
                if not same(last_number(t), r["gold"]):
                    continue
                if len(re.findall(r"^ANSWER:", t, re.M)) != 1:
                    continue
                key = re.sub(r"\s+", " ", re.sub(r"[^0-9+\-*/=.]", "", t))
                if key in seen:
                    continue
                seen.add(key)
                cands.append(t)
            rng.shuffle(cands)
            for t in cands[: args.max_per_question]:
                f.write(json.dumps({
                    "prompt": MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip()),
                    "target": t + EOT,
                    "answer": r["gold"],
                    "src": "rft_self",
                }) + "\n")
                kept += 1
    print(json.dumps({"questions": len(rows), "kept": kept,
                      "keep_rate_per_question": kept / max(1, len(rows))}, indent=2))


if __name__ == "__main__":
    main()
