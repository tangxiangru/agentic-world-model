#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per problem from a checkpoint,
keep the ones whose 'ANSWER: N' line matches gold, emit training rows."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt import load_template, user_prompt  # noqa: E402

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
STOP = "<end_of_turn>"
INT_RE = re.compile(r"^-?\d+$")
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def norm(x: str) -> str | None:
    x = x.replace(",", "").strip()
    try:
        f = float(x)
    except ValueError:
        return None
    if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
        return None
    return str(int(f)) if f == int(f) else str(f)


def collect_problems(n_gsm, n_aug, seed):
    from datasets import load_dataset

    rng = random.Random(seed)
    probs = []
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for r in gsm:
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        if INT_RE.match(final):
            probs.append((r["question"], final, "gsm8k_train"))
    probs = probs[:n_gsm]

    if n_aug > 0:
        omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
        omi = omi.filter(lambda x: x["problem_source"] == "augmented_gsm8k", num_proc=8)
        seen = set()
        aug = []
        for r in omi:
            q = r["problem"]
            if q in seen:
                continue
            seen.add(q)
            e = r["expected_answer"].strip()
            if INT_RE.match(e):
                aug.append((q, e, "augmented_gsm8k"))
        rng.shuffle(aug)
        probs += aug[:n_aug]
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm", type=int, default=7473)
    ap.add_argument("--n-aug", type=int, default=13000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(BASE)
    tpl = load_template()
    probs = collect_problems(args.n_gsm, args.n_aug, args.seed)
    print(f"problems: {len(probs)}", flush=True)

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": user_prompt(q)}],
            chat_template=tpl, tokenize=False, add_generation_prompt=True,
        )
        for q, _, _ in probs
    ]

    llm = LLM(
        model=args.model, gpu_memory_utilization=args.gpu_util,
        max_model_len=4096, dtype="bfloat16", seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    # dump raw generations first: a bug in the filter must never cost the GPU time again
    raw_path = args.out.replace(".jsonl", "_raw.jsonl")
    with open(raw_path, "w") as f:
        for (q, gold, src), o, p in zip(probs, outs, prompts):
            f.write(json.dumps({"q": q, "gold": gold, "src": src, "prompt": p,
                                "samples": [c.text for c in o.outputs]}) + "\n")
    print("raw dumped:", raw_path, flush=True)

    rng = random.Random(args.seed)
    rows, stats = [], {"solved": 0, "unsolved": 0, "n_kept": 0, "gen": 0}
    per_source = {}
    for (q, gold, src), o, p in zip(probs, outs, prompts):
        goldn = norm(gold)
        cands = []
        for c in o.outputs:
            stats["gen"] += 1
            t = c.text.strip()
            m = ANS_RE.search(t)
            if not m:
                continue
            if norm(m.group(1)) != goldn:
                continue
            body = t[: m.start()].rstrip()
            if not body or "ANSWER:" in body:
                continue
            cands.append(f"{body}\n\nANSWER: {goldn}{STOP}")
        cands = list(dict.fromkeys(cands))
        if not cands:
            stats["unsolved"] += 1
            continue
        stats["solved"] += 1
        cands.sort(key=len)  # prefer the shorter correct chains
        keep = cands[: args.keep_per_problem]
        for c in keep:
            rows.append({"prompt": p, "completion": c, "source": f"rft_{src}", "question": q})
            stats["n_kept"] += 1
            per_source[src] = per_source.get(src, 0) + 1

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats["per_source"] = per_source
    stats["pass_at_k"] = stats["solved"] / max(1, stats["solved"] + stats["unsolved"])
    with open(args.out.replace(".jsonl", "_report.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
