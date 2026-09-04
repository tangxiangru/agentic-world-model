#!/usr/bin/env python3
"""Rejection-sampling data generation from a fine-tuned checkpoint.

Samples k solutions per GSM8K-train-derived problem with vLLM, keeps the ones
whose final ANSWER: line matches the gold answer, dedups near-identical chains,
and writes rows in exactly the format train_sft.py consumes.

Only problems from the GSM8K *train* split (and OpenMathInstruct-2's
augmentations of it) are prompted; no test item is read.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

from sft_common import END_OF_TURN, get_tokenizer, render_prompt  # noqa: E402

ANSWER_LINE = re.compile(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)")


def norm(x: str) -> str | None:
    x = str(x).replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return format(float(x), ".5g")
    except ValueError:
        return None


def signature(text: str) -> str:
    """Chain fingerprint: the ordered sequence of numbers it computes."""
    return "|".join(re.findall(r"-?\d+\.?\d*", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="data/rft_pool.jsonl")
    ap.add_argument("--out", default="data/rft_v1.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-problems", type=int, default=-1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    args = ap.parse_args()

    with open(args.pool) as f:
        pool = [json.loads(l) for l in f]
    random.Random(args.seed).shuffle(pool)
    if args.max_problems > 0:
        pool = pool[: args.max_problems]
    print(f"{len(pool)} problems, k={args.k}", flush=True)

    tok = get_tokenizer()
    # pre-tokenised with add_special_tokens=False: the chat template already
    # emits <bos>, and vLLM's LLM.generate() would add a second one for a str prompt
    prompts = [
        {"prompt_token_ids": tok(render_prompt(tok, None, r["user"]), add_special_tokens=False)["input_ids"]}
        for r in pool
    ]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
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
    )
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed + 1)
    rows, n_correct, n_total = [], 0, 0
    per_problem_solved = 0
    for r, o in zip(pool, outs):
        gold = norm(r["answer"])
        seen, kept = set(), []
        for c in o.outputs:
            n_total += 1
            text = c.text.strip()
            m = ANSWER_LINE.search(text)
            if not m or norm(m.group(1)) != gold:
                continue
            n_correct += 1
            # the completion must be exactly one answer, and end with it
            if text.count("ANSWER:") != 1:
                continue
            tail = text[m.end():].strip()
            if tail:
                continue
            sig = signature(text)
            if sig in seen:
                continue
            seen.add(sig)
            kept.append(text)
        if kept:
            per_problem_solved += 1
        for text in kept[: args.keep_per_problem]:
            rows.append(
                {
                    "id": f"rft-{len(rows)}",
                    "system": r.get("system_prefix") if rng.random() < args.fewshot_frac else None,
                    "user": r["user"],
                    "target": text + END_OF_TURN,
                    "answer": r["answer"],
                    "src": "rft:" + r.get("src", "?"),
                    "fewshot": False,
                }
            )

    print(
        f"sampled {n_total}, correct {n_correct} ({n_correct/max(n_total,1):.3f}); "
        f"problems with >=1 correct: {per_problem_solved}/{len(pool)} "
        f"({per_problem_solved/max(len(pool),1):.3f}); kept rows {len(rows)}",
        flush=True,
    )

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "_forcheck.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["user"] + "\n" + r["target"]}) + "\n")
    stats = {
        "model": args.model,
        "n_problems": len(pool),
        "k": args.k,
        "n_samples": n_total,
        "n_correct": n_correct,
        "pass_at_k": per_problem_solved / max(len(pool), 1),
        "sample_accuracy": n_correct / max(n_total, 1),
        "n_rows": len(rows),
    }
    with open(args.out.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
