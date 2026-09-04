#!/usr/bin/env python3
"""Greedy accuracy on a local jsonl probe set, rendered exactly like the grader.

Not a substitute for evaluate.py -- this is the watch-set / diagnostic read on
held-out openai/gsm8k *train* problems (rule 7: never the benchmark test copy).
Scoring reproduces inspect_ai's match(numeric=True, location="end"): the last
numeric word of the completion, punctuation-stripped and normalised.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402


def last_number(text: str) -> str | None:
    v = text.strip().casefold()
    v = re.sub(r"[,$%]", "", v)
    words = re.split(r"\s+", v)
    words.reverse()
    for w in words:
        w = w.strip(".!?:;()[]{}'\"")
        if w.replace(".", "").replace("-", "").isnumeric():
            return w.lstrip("-") if w.startswith("--") else w
    return None


def norm(x: str) -> str:
    try:
        return format(float(x), ".5g")
    except ValueError:
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--nshot", type=int, default=10,
                    help="few-shot exemplars, matching the grader's 10")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.probe)]
    if args.limit:
        items = items[: args.limit]

    fs = None
    if args.nshot:
        import datasets
        import random
        tr = datasets.load_dataset("openai/gsm8k", "main", split="train")
        rng = random.Random(42)
        idxs = rng.sample(range(len(tr)), args.nshot)
        blocks = []
        for i in idxs:
            r = tr[i]
            parts = r["answer"].split("####")
            blocks.append(render.fewshot_block(r["question"], "####".join(parts[:-1]).strip(),
                                               parts[-1].strip()))
        fs = blocks

    prompts = [render.build_prompt(it["question"], fs) for it in items]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, enforce_eager=False, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    rows, ncorr, ncap = [], 0, 0
    for it, o in zip(items, outs):
        txt = o.outputs[0].text
        pred = last_number(txt)
        ok = pred is not None and norm(pred) == norm(it["answer"])
        ncorr += ok
        capped = o.outputs[0].finish_reason == "length"
        ncap += capped
        rows.append({"question": it["question"], "gold": it["answer"], "pred": pred,
                     "correct": bool(ok), "capped": bool(capped), "text": txt})
    summary = {"model": args.model, "probe": args.probe, "n": len(rows),
               "accuracy": ncorr / len(rows), "capped": ncap, "nshot": args.nshot}
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
