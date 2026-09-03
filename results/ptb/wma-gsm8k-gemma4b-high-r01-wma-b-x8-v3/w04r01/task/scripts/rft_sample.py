"""Rejection sampling: draw k solutions per training question from a checkpoint,
keep the ones whose ANSWER line matches gold, write a new prompt/completion jsonl.

Prompts and targets go through scripts/fmt.py, so what is sampled is exactly what the
grader would see and what is written back is exactly what the trainer reads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

from datasets import load_dataset, load_from_disk  # noqa: E402

CALC = re.compile(r"<<[^>]*>>")
ANS = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def norm(x: str | None) -> str | None:
    if x is None:
        return None
    x = x.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(x)
    except ValueError:
        return None
    return str(int(f)) if f.is_integer() else str(f)


def extract(text: str) -> str | None:
    ms = ANS.findall(text)
    return norm(ms[-1]) if ms else None


def gsm8k_questions():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        gold = norm(r["answer"].rpartition("####")[2])
        if gold is not None:
            yield r["question"], gold


def omi2_questions(path, n):
    ds = load_from_disk(path)
    seen = set()
    for r in ds:
        if r["problem_source"] != "augmented_gsm8k":
            continue
        q = r["problem"]
        if q in seen:
            continue
        seen.add(q)
        gold = norm(r["expected_answer"])
        if gold is None:
            continue
        yield q, gold
        if len(seen) >= n:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-q", type=int, default=2)
    ap.add_argument("--n-omi2", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--prefer", choices=["shortest", "longest"], default="shortest",
                    help="which correct chains to keep when more than keep-per-q survive")
    ap.add_argument("--kshot", action="store_true",
                    help="render output prompts with the same k-shot mix exp-03 trained on")
    args = ap.parse_args()

    import random as _random

    from build_fewshot_data import fewshot_pool, shot_block

    rng = _random.Random(args.seed)
    pool = fewshot_pool() if args.kshot else None
    KS, WS = [0, 3, 6, 10], [0.30, 0.10, 0.15, 0.45]

    def out_prompt(q: str) -> str:
        if not args.kshot:
            return fmt.render_prompt(q)
        k = rng.choices(KS, WS)[0]
        if not k:
            return fmt.render_prompt(q)
        shots = [s for s in rng.sample(pool, k + 2) if s[0] != q][:k]
        return fmt.render_prompt(q, fewshot_system="\n\n".join(shot_block(*s) for s in shots))

    qs = list(gsm8k_questions())
    n_gsm = len(qs)
    if args.n_omi2:
        qs += list(omi2_questions("/home/ben/task/data/omi2_1M", args.n_omi2))
    print(f"{len(qs)} questions ({n_gsm} gsm8k train, {len(qs)-n_gsm} omi2)", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=2048,
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    prompts = [fmt.render_prompt(q) for q, _ in qs]
    outs = llm.generate(prompts, sp)

    kept, per_q = [], defaultdict(int)
    n_any_correct = n_samples_correct = n_samples = 0
    solved_flags = []
    for (q, gold), o in zip(qs, outs):
        texts = [c.text for c in o.outputs]
        n_samples += len(texts)
        good, seen_bodies = [], set()
        for t in texts:
            if extract(t) == gold:
                n_samples_correct += 1
                body = t.strip()
                # one answer marker only; cut anything after the ANSWER line
                idx = body.rfind(fmt.ANSWER_MARKER)
                body = body[:idx].strip()
                if not body:
                    continue
                key = re.sub(r"\s+", " ", body.lower())
                if key in seen_bodies:
                    continue
                seen_bodies.add(key)
                good.append(body)
        solved_flags.append(bool(good))
        if good:
            n_any_correct += 1
            # exp-05 kept the shortest chains and the model got terser and no better;
            # 'longest' keeps the most step-by-step derivation instead.
            good.sort(key=len, reverse=(args.prefer == "longest"))
            for body in good[: args.keep_per_q]:
                kept.append(
                    {
                        "prompt": out_prompt(q),
                        "completion": fmt.render_completion(body, gold),
                        "source": "rft_self",
                        "question": q,
                        "answer": gold,
                    }
                )
                per_q[q] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {
        "n_questions": len(qs),
        "k": args.k,
        "temp": args.temp,
        "pass_at_k": round(n_any_correct / len(qs), 4),
        "sample_accuracy": round(n_samples_correct / max(1, n_samples), 4),
        "n_kept": len(kept),
        "n_questions_with_data": len(per_q),
        "out": args.out,
    }
    print(json.dumps(stats, indent=1), flush=True)
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
