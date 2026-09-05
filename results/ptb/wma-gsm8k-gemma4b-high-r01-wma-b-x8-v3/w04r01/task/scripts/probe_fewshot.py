"""Diagnostic probe: does the grader's 10-shot prefix change behaviour vs zero-shot?

The model is trained zero-shot but always graded with a 10-shot system prefix built from
openai/gsm8k split=train (seed 42, shuffled) - exactly what inspect_evals.gsm8k does.  This
runs the same held-out questions both ways and reports accuracy and the share of completions
that do not end in an ANSWER line.  No test item is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402
from rft_sample import extract, norm  # noqa: E402

from datasets import load_dataset  # noqa: E402

ANSWER_LINE = re.compile(r"^ANSWER:\s*\$?-?[\d,]+(?:\.\d+)?\.?$")


def build_fewshot_system() -> str:
    """Byte-identical to inspect_evals.gsm8k: hf_dataset(shuffle=True, seed=42, limit=10)."""
    from inspect_ai.dataset import hf_dataset

    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=10,
    )
    return "\n\n".join([sample_to_fewshot(s) for s in fewshots])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=7000)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    items = []
    for r in ds.select(range(args.offset, min(len(ds), args.offset + args.n))):
        g = norm(r["answer"].rpartition("####")[2])
        if g is not None:
            items.append((r["question"], g))

    sysmsg = build_fewshot_system()
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac, max_model_len=4096)
    sp = SamplingParams(temperature=args.temp, max_tokens=1024, stop_token_ids=[1, 106])

    res = {}
    for mode, sysm in (("zero_shot", None), ("ten_shot", sysmsg)):
        prompts = [fmt.render_prompt(q, fewshot_system=sysm) for q, _ in items]
        outs = llm.generate(prompts, sp)
        n_ok = n_badfmt = 0
        lens = []
        for (q, gold), o in zip(items, outs):
            t = o.outputs[0].text
            lens.append(len(t))
            if extract(t) == gold:
                n_ok += 1
            last = t.strip().split("\n")[-1].strip()
            if not ANSWER_LINE.match(last):
                n_badfmt += 1
        res[mode] = {
            "n": len(items),
            "accuracy": round(n_ok / len(items), 4),
            "bad_format_share": round(n_badfmt / len(items), 4),
            "mean_chars": round(sum(lens) / len(lens), 1),
        }
        print(mode, json.dumps(res[mode]), flush=True)

    res["model"] = args.model
    res["items_from"] = f"openai/gsm8k train[{args.offset}:{args.offset+args.n}]"
    res["temperature"] = args.temp
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
