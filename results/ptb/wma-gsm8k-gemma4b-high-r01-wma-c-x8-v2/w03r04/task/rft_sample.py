#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per training question
from a checkpoint, keep the ones whose final number matches the gold answer.

Questions come only from the GSM8K TRAIN split (minus the held-out probe) and
from OpenMathInstruct-2's gsm8k-derived augmented problems, which are
themselves built from GSM8K train. The test split is never read.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from train_sft import BASE, MATH_PROMPT_TEMPLATE, fewshot_block

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str):
    m = NUM_RE.findall(text.replace("$", ""))
    if not m:
        return None
    v = m[-1].replace(",", "").rstrip(".")
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft_eot.jsonl")
    ap.add_argument("--n-questions", type=int, default=12000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--p-fewshot", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probe_q = {json.loads(l)["question"].strip() for l in open("data/probe200.jsonl")}
    split = json.load(open("data/split_idx.json"))
    g = load_dataset("openai/gsm8k", "main")["train"]

    qs = []
    for i in sorted(split["train_idx"]):
        r = g[i]
        if r["question"].strip() in probe_q:
            continue
        qs.append((r["question"].strip(), r["answer"].rsplit("####", 1)[1].strip()))
    n_human = len(qs)

    if args.n_questions > n_human:
        omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        omi = omi.filter(
            lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8
        )
        pool = list(range(len(omi)))
        rng.shuffle(pool)
        seen = {q for q, _ in qs}
        for i in pool:
            if len(qs) >= args.n_questions:
                break
            r = omi[i]
            q = r["problem"].strip()
            if q in seen or q in probe_q:
                continue
            seen.add(q)
            qs.append((q, r["expected_answer"].strip()))
    qs = qs[: args.n_questions]
    print(f"{len(qs)} questions ({n_human} human gsm8k-train)")

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("templates/gemma3.jinja").read()

    fewshot_pool = []
    for i in split["train_idx"][:2000]:
        r = g[i]
        b, a = r["answer"].rsplit("####", 1)
        fewshot_pool.append((r["question"], b.strip(), a.strip()))

    prompts = []
    for q, _ in qs:
        msgs = []
        if args.p_fewshot and rng.random() < args.p_fewshot:
            msgs.append(
                {"role": "system", "content": fewshot_block(rng.sample(fewshot_pool, 4))}
            )
        msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=2048,
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

    kept, n_correct, n_total = [], 0, 0
    for (q, gold), o in zip(qs, outs):
        try:
            goldf = float(gold.replace(",", ""))
        except ValueError:
            continue
        cands = []
        for c in o.outputs:
            n_total += 1
            t = c.text.strip()
            if c.finish_reason != "stop":
                continue
            lines = [l for l in t.split("\n") if l.strip()]
            if not lines or not re.match(r"^ANSWER:\s*[-\d,.$]+\s*$", lines[-1].strip()):
                continue
            v = last_number(t)
            if v is None or abs(v - goldf) > 1e-6:
                continue
            n_correct += 1
            cands.append(t)
        # difficulty-aware: an item the model already solves every time earns one
        # sample; the ones it only sometimes solves earn the full quota
        uniq = sorted(set(cands), key=len)
        quota = 1 if len(cands) == args.k else args.max_per_question
        for t in uniq[:quota]:
            kept.append({"question": q, "target": t + "<end_of_turn>", "source": "rft_self", "answer": gold})

    print(f"{n_correct}/{n_total} samples correct ({n_correct/max(1,n_total):.3f}); kept {len(kept)} rows")
    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
