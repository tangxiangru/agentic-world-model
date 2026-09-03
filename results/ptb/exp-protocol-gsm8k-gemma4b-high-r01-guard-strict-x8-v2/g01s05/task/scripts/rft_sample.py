#!/usr/bin/env python3
"""Rejection sampling: draw k chains per GSM8K *train* question from a fine-tuned
checkpoint, keep the ones whose final answer matches gold, dedup, and write an SFT
jsonl in the same rendered {prompt, completion} form as build_data.py.

No test item is touched: questions come from openai/gsm8k split=train only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

TASK = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK, "templates", "gemma3.jinja")

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANSWER_MARKER = "ANSWER: "
STOP_TOKEN = "<end_of_turn>"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    ms = NUM.findall(text)
    if not ms:
        return None
    s = ms[-1].replace(",", "").rstrip(".")
    try:
        v = float(s)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, OverflowError):
        return None


def norm_answer(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if re.fullmatch(r"-?\d+", s):
        return str(int(s))
    return None


def equation_signature(sol: str) -> str:
    """Chains that perform the same arithmetic in the same order are duplicates."""
    eqs = re.findall(r"[-+*/=0-9().]{3,}", sol)
    return hashlib.sha1("|".join(e.replace(" ", "") for e in eqs).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-per-q", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(TEMPLATE_PATH) as f:
        tok.chat_template = f.read()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    qs = []
    for r in ds:
        a = norm_answer(r["answer"].split("####")[-1])
        if a is None:
            continue
        qs.append((r["question"].strip(), a))
    if args.limit:
        qs = qs[: args.limit]
    print("questions:", len(qs), flush=True)

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q, _ in qs
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=2048,
        dtype="bfloat16",
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_new,
        stop=[STOP_TOKEN],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    # dump raw generations first: filtering must never be able to lose 10 GPU-minutes
    raw_path = args.out.replace(".jsonl", ".raw.jsonl")
    with open(raw_path, "w") as f:
        for (q, gold), o, prompt in zip(qs, outs, prompts):
            f.write(json.dumps({"q": q, "gold": gold, "prompt": prompt,
                                "gens": [c.text for c in o.outputs]}) + "\n")
    print("raw dumped to", raw_path, flush=True)

    kept, n_corr, n_tot = [], 0, 0
    per_q_solved = 0
    for (q, gold), o, prompt in zip(qs, outs, prompts):
        sigs = set()
        picked = 0
        any_ok = False
        for c in o.outputs:
            n_tot += 1
            text = c.text.strip()
            if ANSWER_MARKER not in text:
                continue
            pred = last_number(text)
            if pred != gold:
                continue
            n_corr += 1
            any_ok = True
            # exactly one answer marker, ends with the marker line
            if text.count(ANSWER_MARKER) != 1:
                continue
            head, _, tail = text.rpartition(ANSWER_MARKER)
            if tail.strip() != gold:
                continue
            body = head.strip()
            if len(body) < 20 or "####" in body or "\\boxed" in body:
                continue
            sig = equation_signature(body)
            if sig in sigs:
                continue
            sigs.add(sig)
            picked += 1
            kept.append(
                {
                    "prompt": prompt,
                    "completion": f"{body}\n\n{ANSWER_MARKER}{gold}{STOP_TOKEN}",
                    "src": "rft",
                    "question": q,
                }
            )
            if picked >= args.max_per_q:
                break
        per_q_solved += int(any_ok)

    print(
        f"samples {n_tot}, correct {n_corr} ({n_corr/max(n_tot,1):.3f}), "
        f"questions with >=1 correct {per_q_solved}/{len(qs)} ({per_q_solved/len(qs):.3f}), "
        f"kept {len(kept)}",
        flush=True,
    )
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", ".contam.jsonl"), "w") as f:
        for r in kept:
            f.write(json.dumps({"text": r["prompt"] + r["completion"]}) + "\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
