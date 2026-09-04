#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per training question from a checkpoint,
keep only those whose final 'ANSWER: N' matches the gold answer, dedup, write SFT rows.

Questions come from GSM8K TRAIN and from OpenMathInstruct-2 problems (GSM8K/MATH train
derived) — never from the benchmark test set.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from build_data import EOT, render_prompt, sid

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def norm(x: str) -> float | None:
    try:
        return float(x.replace(",", "").rstrip("."))
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with {question, gold}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--keep-per-question-hard", type=int, default=4)
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(args.questions)]
    if args.max_questions:
        random.Random(args.seed).shuffle(qs)
        qs = qs[: args.max_questions]
    print(f"[rft] {len(qs)} questions x k={args.k}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=3072,
        seed=args.seed,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    prompts = [render_prompt(q["question"]) for q in qs]
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    n_kept = 0
    n_solved = 0
    pass_at_k = 0
    with open(args.out, "w") as f:
        for q, o in zip(qs, outs):
            gold = norm(str(q["gold"]))
            texts, any_ok = [], False
            for c in o.outputs:
                t = c.text.strip()
                m = ANS_RE.search(t)
                if not m:
                    continue
                v = norm(m.group(1))
                if v is None or gold is None or abs(v - gold) > 1e-6:
                    continue
                # keep only the part up to and including the answer line
                end = m.end()
                t = t[:end].strip()
                if t.count("ANSWER:") != 1:
                    continue
                any_ok = True
                texts.append(t)
            pass_at_k += any_ok
            # dedup identical samples; keep more paths for the questions the model
            # finds hard (pass rate <= 0.5), fewer for the ones it already nails
            uniq = sorted(set(texts))
            rng.shuffle(uniq)
            cap = args.keep_per_question
            if len(texts) <= args.k / 2:
                cap = args.keep_per_question_hard
            for t in uniq[:cap]:
                prompt = render_prompt(q["question"])
                target = f"{t}{EOT}"
                f.write(
                    json.dumps(
                        {
                            "id": sid(prompt + target),
                            "source": "rft:self",
                            "prompt": prompt,
                            "target": target,
                            "gold": str(q["gold"]),
                        }
                    )
                    + "\n"
                )
                n_kept += 1
            n_solved += len(uniq) > 0
    stats = {
        "n_questions": len(qs),
        "k": args.k,
        "pass_at_k": pass_at_k / len(qs),
        "questions_with_a_kept_sample": n_solved / len(qs),
        "rows_written": n_kept,
        "model": args.model,
    }
    print(json.dumps(stats, indent=2), flush=True)
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
