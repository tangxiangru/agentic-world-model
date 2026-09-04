#!/usr/bin/env python3
"""Offline vLLM generation used for (a) a fast internal dev score and
(b) rejection-sampling data generation. Prompts are rendered with the grader's
own template and scored with the grader's own last-number rule.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from render import build_fewshot_system_text, render_prompt


def last_number(text: str) -> str | None:
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w2 = re.sub(r"[$,£€*_]", "", w)
        w2 = re.sub(r"\.(?=\s|$|\D)", "", w2)
        if w2.replace(".", "").isnumeric():
            return w2
    return None


def norm(x: str | None) -> str | None:
    if x is None:
        return None
    x = x.lstrip("0") or "0"
    if "." in x:
        x = x.rstrip("0").rstrip(".") or "0"
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with {question, answer}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="samples per question")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--fewshot", type=int, default=10, help="0 = no system message")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    system = build_fewshot_system_text() if args.fewshot else None
    prompts = [render_prompt(r["question"], system) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=4096,
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
    )
    outs = llm.generate(prompts, sp)

    n_correct = 0
    n_any = 0
    with open(args.out, "w") as f:
        for row, out in zip(rows, outs):
            gold = norm(str(row["answer"]).strip().replace(",", ""))
            comps = [o.text for o in out.outputs]
            oks = [norm(last_number(c)) == gold for c in comps]
            n_correct += int(oks[0])
            n_any += int(any(oks))
            f.write(
                json.dumps(
                    {
                        "question": row["question"],
                        "answer": gold,
                        "completions": comps,
                        "correct": oks,
                    }
                )
                + "\n"
            )
    n = len(rows)
    summary = {
        "model": args.model,
        "n": n,
        "samples_per_q": args.n,
        "temperature": args.temperature,
        "fewshot": args.fewshot,
        "acc_first_sample": n_correct / n,
        "pass_at_n": n_any / n,
    }
    print(json.dumps(summary, indent=1))
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
