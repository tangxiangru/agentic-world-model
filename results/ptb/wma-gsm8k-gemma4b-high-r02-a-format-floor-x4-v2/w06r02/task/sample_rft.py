#!/usr/bin/env python3
"""Rejection-sampling data build: sample k solutions per training problem from a
checkpoint, keep the ones whose final 'ANSWER: <n>' matches the gold answer.

Problems come from the GSM8K *train* split and/or the OpenMathInstruct-2 gsm8k-derived
pool already built in data/sft_gsm8k.jsonl. The test split is never touched.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\.?\s*$")


def norm(x: str):
    try:
        return round(float(str(x).replace(",", "").replace("$", "").strip()), 4)
    except Exception:
        return None


def get_answer(text: str):
    m = ANS_RE.search(text.strip())
    return norm(m.group(1)) if m else None


def load_problems(args):
    probs = {}
    if args.source in ("gsm8k", "both"):
        from datasets import load_dataset

        for r in load_dataset("openai/gsm8k", "main", split="train"):
            gold = norm(r["answer"].split("####")[-1])
            if gold is not None:
                probs[r["question"].strip()] = gold
    if args.source in ("omi", "both"):
        for line in open("data/sft_gsm8k.jsonl"):
            d = json.loads(line)
            g = norm(d["answer"])
            if g is not None:
                probs.setdefault(d["question"].strip(), g)
    items = sorted(probs.items())
    random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="gsm8k", choices=["gsm8k", "omi", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--kshot-min", type=int, default=0)
    ap.add_argument("--kshot-max", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    template = open("templates/gemma3.jinja").read()
    tok = AutoTokenizer.from_pretrained(args.model)
    items = load_problems(args)
    print(f"[rft] {len(items)} problems, k={args.k}", flush=True)

    # optional k-shot prefix, built exactly as inspect_evals.gsm8k.sample_to_fewshot does,
    # from the GSM8K *train* split - this mirrors the prompt the grader actually sends.
    shots = []
    if args.kshot_max:
        from datasets import load_dataset

        for r in load_dataset("openai/gsm8k", "main", split="train"):
            reasoning, target = r["answer"].split("####")
            shots.append(f"{r['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {target.strip()}")

    rng = random.Random(args.seed)
    user_texts = []
    for q, _ in items:
        base = PROMPT_TEMPLATE.format(prompt=q)
        if args.kshot_max:
            k = rng.randint(args.kshot_min, args.kshot_max)
            if k:
                base = "\n\n".join(rng.sample(shots, k)) + "\n\n" + base
        user_texts.append(base)

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": u}],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
        )
        for u in user_texts
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    rows, per_problem_correct = [], []
    for (q, gold), u, o in zip(items, user_texts, outs):
        cands = []
        seen = set()
        for c in o.outputs:
            txt = c.text.strip()
            if get_answer(txt) == gold and txt not in seen:
                seen.add(txt)
                cands.append(txt)
        per_problem_correct.append(len(cands))
        cands.sort(key=len)  # prefer the more concise correct chains
        for txt in cands[: args.keep_per_problem]:
            rows.append(
                {
                    "prompt": u,
                    "completion": txt,
                    "question": q,
                    "answer": str(gold),
                    "source": "rft_self",
                }
            )

    random.Random(args.seed).shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    solved = sum(1 for c in per_problem_correct if c > 0)
    stats = {
        "problems": len(items),
        "k": args.k,
        "rows_kept": len(rows),
        "pass_at_k": solved / max(1, len(items)),
        "mean_correct_per_problem": sum(per_problem_correct) / max(1, len(items)),
        "unsolved": len(items) - solved,
    }
    print("[rft]", json.dumps(stats), flush=True)
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
