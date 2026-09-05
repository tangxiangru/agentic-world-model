#!/usr/bin/env python3
"""Cheap internal eval on held-out GSM8K *train* problems (never the test split).

Reports accuracy under the prompt shapes and decoding settings we care about, so
prompt/decoding decisions can be made without spending benchmark queries.
"""
from __future__ import annotations

import argparse
import json
import random
import re

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.MULTILINE)


def norm(s):
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


def score(text, gold):
    """Mirror inspect's match(numeric=True, location='end'): last numeric token."""
    words = text.strip().replace(",", "").replace("$", "").split()
    words.reverse()
    for w in words:
        w2 = w.strip(".:;!?()")
        if w2.replace(".", "").isnumeric():
            return norm(w2) == gold
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--holdout-start", type=int, default=6900)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = list(range(args.holdout_start, min(args.holdout_start + args.n, len(ds))))
    probs = [(ds[i]["question"], norm(ds[i]["answer"].split("####")[-1])) for i in idx]

    # 10-shot prefix built the same way the harness does (from the train split)
    rng = random.Random(42)
    shot_rows = [ds[i] for i in rng.sample(range(0, 6000), 10)]
    shots = []
    for r in shot_rows:
        reasoning, _, final = r["answer"].partition("####")
        reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
        shots.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {final.strip()}")
    prefix10 = "\n\n".join(shots) + "\n\n"

    def mk(q, prefix=""):
        return f"<bos><start_of_turn>user\n{prefix}{MATH_PROMPT_TEMPLATE.format(prompt=q.strip())}<end_of_turn>\n<start_of_turn>model\n"

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=3072,
              enable_prefix_caching=True, dtype="bfloat16")

    settings = {
        "greedy": SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["<end_of_turn>"]),
        "sampled_t1_p095_k64": SamplingParams(temperature=1.0, top_p=0.95, top_k=64, seed=0,
                                              max_tokens=args.max_tokens, stop=["<end_of_turn>"]),
    }
    shapes = {"zeroshot": "", "tenshot": prefix10}

    results = {}
    for sname, sp in settings.items():
        for pname, pfx in shapes.items():
            prompts = [mk(q, pfx) for q, _ in probs]
            outs = llm.generate(prompts, sp)
            acc = sum(score(o.outputs[0].text, g) for o, (_, g) in zip(outs, probs)) / len(probs)
            trunc = sum(o.outputs[0].finish_reason != "stop" for o in outs) / len(probs)
            ntok = sum(len(o.outputs[0].token_ids) for o in outs) / len(probs)
            results[f"{sname}/{pname}"] = {"acc": round(acc, 4), "trunc": round(trunc, 4), "mean_tokens": round(ntok, 1)}
            print(f"{sname:22s} {pname:9s} acc={acc:.4f} trunc={trunc:.3f} tok={ntok:.0f}", flush=True)

    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
