#!/usr/bin/env python3
"""Diagnostic: does the grader's fixed 10-shot prefix cost the SFT model
accuracy relative to the 0-shot prompt it was mostly trained on?

Runs on held-out OpenMathInstruct-2 GSM8K-derived problems (never the GSM8K
test split), greedy, both prompt conditions, same items.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

OMI2 = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
    "469216e3f46f4dacf476b382e192485ea51a143e/data/train-%05d-of-00032.parquet"
)
NUMERIC = re.compile(r"^-?\d{1,12}(\.\d{1,6})?$")
PREFIX = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_10shot_prefix.txt")).read()


def norm(x: str) -> str:
    x = x.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=20)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--exclude", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seen = set()
    if args.exclude and os.path.exists(args.exclude):
        for line in open(args.exclude):
            seen.add(json.loads(line)["question"].strip()[:200])

    probs = []
    for r in pq.read_table(OMI2 % args.shard).to_pylist():
        if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
            continue
        q, a = r["problem"].strip(), (r["expected_answer"] or "").strip()
        if not NUMERIC.match(a) or q[:200] in seen:
            continue
        seen.add(q[:200])
        probs.append({"question": q, "answer": a})
        if len(probs) >= args.n:
            break
    print(f"[probe] {len(probs)} held-out problems", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=4096,
              dtype="bfloat16", seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=640, stop=[fmt.STOP_TOKEN])

    res = {}
    for name, system in (("zero_shot", None), ("ten_shot", PREFIX)):
        prompts = [fmt.render_prompt(p["question"], system) for p in probs]
        outs = llm.generate(prompts, sp)
        ok = 0
        for p, o in zip(probs, outs):
            t = o.outputs[0].text
            m = re.search(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)", t)
            ok += bool(m) and norm(m.group(1)) == norm(p["answer"])
        res[name] = ok / len(probs)
        print(f"[probe] {name}: {ok}/{len(probs)} = {res[name]:.4f}", flush=True)
    res["n"] = len(probs)
    res["delta_tenshot_minus_zeroshot"] = res["ten_shot"] - res["zero_shot"]
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
