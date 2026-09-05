#!/usr/bin/env python3
"""Probe a checkpoint on held-out GSM8K-style problems, with and without the grader's
10-shot prefix. Measures accuracy and how often the model falls into the terse
'<<a*b=c>>' GSM8K-train style. No benchmark test item is read.
"""

from __future__ import annotations

import argparse
import json
import os
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import fmt  # noqa: E402

ANSWER_RE = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)")


def norm(a: str) -> str:
    a = a.strip().replace(",", "").replace("$", "")
    return a[:-2] if a.endswith(".0") else a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default="/home/ben/task/data/probe300.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=768)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    items = [json.loads(l) for l in open(args.probe)]
    tok = AutoTokenizer.from_pretrained(fmt.SNAPSHOT)
    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=4096, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=[1, 106])

    report = {}
    for mode in ("fewshot", "zeroshot"):
        ids = [
            tok(fmt.render_prompt(it["question"], fewshot=(mode == "fewshot")),
                add_special_tokens=False)["input_ids"]
            for it in items
        ]
        outs = llm.generate([{"prompt_token_ids": i} for i in ids], sampling_params=sp)
        ok = terse = trunc = 0
        terse_ok = verb_ok = terse_n = verb_n = 0
        for it, o in zip(items, outs):
            t = o.outputs[0].text
            m = ANSWER_RE.search(t)
            good = bool(m) and norm(m.group(1)) == norm(it["answer"])
            ok += good
            is_terse = "<<" in t
            terse += is_terse
            trunc += o.outputs[0].finish_reason == "length"
            if is_terse:
                terse_n += 1
                terse_ok += good
            else:
                verb_n += 1
                verb_ok += good
        report[mode] = {
            "n": len(items),
            "accuracy": ok / len(items),
            "terse_frac": terse / len(items),
            "truncated": trunc,
            "terse_acc": terse_ok / terse_n if terse_n else None,
            "verbose_acc": verb_ok / verb_n if verb_n else None,
        }
        print(mode, json.dumps(report[mode]))
    json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
