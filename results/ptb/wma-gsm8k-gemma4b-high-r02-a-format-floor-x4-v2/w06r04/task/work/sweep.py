"""Probe sweep: for each checkpoint, score the 200 held-out TRAIN rows under
both decoders in one vLLM process.

  sampled = the parent's generation_config (T=1.0, top_k 64, top_p 0.95) - the
            decoder the official comparator ran under
  greedy  = temperature 0, top_p 1, top_k off

Same prompts, same template, same scorer as work/probe_eval.py, so numbers are
comparable to the probe readings already in the cards.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_eval import fewshot_system_message, render, user_message, grade  # noqa: E402

DECODES = {
    "sampled": dict(temperature=1.0, top_p=0.95, top_k=64),
    "greedy": dict(temperature=0.0, top_p=1.0, top_k=-1),
}


def score(outs, rows):
    n_ok = n_clean = n_trunc = 0
    for r, o in zip(rows, outs):
        c = o.outputs[0]
        ok = grade(c.text, r["gold"])
        n_ok += ok
        lines = [x for x in c.text.rstrip().split("\n") if x.strip()]
        n_clean += int(c.finish_reason == "stop" and lines and lines[-1].strip().startswith("ANSWER:"))
        n_trunc += int(c.finish_reason == "length")
    n = len(rows)
    return {"accuracy": n_ok / n, "clean_format_rate": n_clean / n,
            "truncation_rate": n_trunc / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--decodes", nargs="+", default=["sampled", "greedy"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_set.jsonl"))][: args.n]
    sysmsg = fewshot_system_message()
    prompts = [render([{"role": "system", "content": sysmsg},
                       {"role": "user", "content": user_message(r["question"])}])
               for r in rows]

    from vllm import LLM, SamplingParams
    import torch

    results = {}
    for m in args.models:
        llm = LLM(model=m, gpu_memory_utilization=0.85, max_model_len=4096,
                  dtype="bfloat16", disable_log_stats=True)
        results[m] = {}
        for d in args.decodes:
            sp = SamplingParams(max_tokens=args.max_tokens, **DECODES[d])
            results[m][d] = score(llm.generate(prompts, sp), rows)
            print(m, d, json.dumps(results[m][d]), flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps(results, indent=2))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
