#!/usr/bin/env python3
"""Offline dev evaluation on held-out OpenMathInstruct-2 GSM8K-derived problems.

Reproduces the grader's prompt (templates/gemma3.jinja + the inspect_evals
MATH_PROMPT_TEMPLATE + an optional k-shot system block of GSM8K TRAIN items) and
the grader's scoring rule (match(numeric=True, location="end"): the LAST number
in the completion must equal gold). Touches no GSM8K test item.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from build_data import MATH_PROMPT_TEMPLATE, render, sample_to_fewshot


def strip_numeric_punctuation(s: str) -> str:
    # inspect_ai._util.text.strip_numeric_punctuation, close enough for scoring
    s = re.sub(r"[$,]", "", s)
    return s


def graded_answer(text: str) -> str | None:
    v = strip_numeric_punctuation(text.strip().casefold())
    words = re.split(r"\s+", v)
    words.reverse()
    for w in words:
        w2 = w.strip(".:;!?()[]{}'\"")
        if w2.replace(".", "").replace("-", "").isnumeric():
            try:
                return format(float(w2), ".5g")
            except ValueError:
                return None
    return None


def norm(t: str) -> str:
    return format(float(strip_numeric_punctuation(t.strip())), ".5g")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/dev_fresh_5014.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--fewshot", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    rows = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            m = re.search(r"ANSWER: ([^\n<]+)", r["completion"])
            if not m:
                continue
            rows.append((r["question"], m.group(1).strip()))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]

    system = None
    if args.fewshot:
        train = load_dataset("openai/gsm8k", "main", split="train")
        idx = list(range(len(train)))
        random.Random(42).shuffle(idx)
        demos = []
        for i in idx[: args.fewshot]:
            parts = train[i]["answer"].split("####")
            demos.append(
                (train[i]["question"], "####".join(parts[:-1]).strip(), parts[-1].strip())
            )
        system = "\n\n".join(sample_to_fewshot(*d) for d in demos)

    prompts = [render(system, MATH_PROMPT_TEMPLATE.format(prompt=q)) for q, _ in rows]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.samples,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    n_ok = n_vote = n_trunc = 0
    recs = []
    for (q, gold), o in zip(rows, outs):
        g = norm(gold)
        answers = [graded_answer(c.text) for c in o.outputs]
        ok = answers[0] == g
        n_ok += ok
        if len(answers) > 1:
            from collections import Counter

            cnt = Counter(a for a in answers if a is not None)
            maj = cnt.most_common(1)[0][0] if cnt else None
            n_vote += maj == g
        n_trunc += o.outputs[0].finish_reason == "length"
        recs.append({"q": q, "gold": g, "pred": answers[0], "ok": bool(ok), "text": o.outputs[0].text})

    res = {
        "model": args.model,
        "n": len(rows),
        "fewshot": args.fewshot,
        "temperature": args.temperature,
        "accuracy": n_ok / len(rows),
        "maj_accuracy": (n_vote / len(rows)) if args.samples > 1 else None,
        "truncated": n_trunc / len(rows),
    }
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": res, "records": recs}, f)


if __name__ == "__main__":
    main()
