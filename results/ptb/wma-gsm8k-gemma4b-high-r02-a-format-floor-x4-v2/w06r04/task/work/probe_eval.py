"""Fast offline probe: same prompt, same template, same scorer as the grader.

Usage:
  python work/probe_eval.py --model <path> [--n 200] [--temperature 1.0]
                            [--top-p 0.95] [--top-k 64] [--out logs/x.json]

Defaults reproduce the grader's decoder for the base checkpoint
(generation_config.json: do_sample, top_k 64, top_p 0.95, temperature unset -> 1.0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_eval import fewshot_system_message, user_message, render, grade  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_set.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--k-samples", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.probe)][: args.n]
    sysmsg = fewshot_system_message()
    prompts = [
        render([{"role": "system", "content": sysmsg},
                {"role": "user", "content": user_message(r["question"])}])
        for r in rows
    ]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=False,
        disable_log_stats=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.k_samples,
    )  # no explicit stop list: vLLM derives stop_token_ids from the model's own
       # generation_config eos_token_id ([1, 106] for gemma-3), exactly as the
       # served grader does. An explicit stop list would make this proxy more
       # forgiving of non-termination than the real harness.
    outs = llm.generate(prompts, sp)

    n_correct = 0
    n_clean = 0
    records = []
    for r, o in zip(rows, outs):
        comps = [c.text for c in o.outputs]
        oks = [grade(c, r["gold"]) for c in comps]
        ok = oks[0]
        n_correct += int(ok)
        # "clean" the way the grader cuts: the completion must stop at eos and its
        # last non-empty line must be the ANSWER line (a number after it is fatal).
        finished = o.outputs[0].finish_reason == "stop"
        lines = [x for x in comps[0].rstrip().split("\n") if x.strip()]
        last = lines[-1].strip() if lines else ""
        clean = finished and last.startswith("ANSWER:")
        n_clean += int(clean)
        records.append({"id": r["id"], "gold": r["gold"], "correct": ok,
                        "clean_format": clean, "finish_reason": o.outputs[0].finish_reason,
                        "n_gen_tokens": len(o.outputs[0].token_ids),
                        "completions": comps, "any_correct": any(oks)})

    acc = n_correct / len(rows)
    pass_k = sum(r["any_correct"] for r in records) / len(rows)
    trunc = sum(1 for r in records if r["finish_reason"] == "length") / len(rows)
    summary = {"model": args.model, "n": len(rows), "accuracy": acc,
               "clean_format_rate": n_clean / len(rows), "truncation_rate": trunc,
               "k_samples": args.k_samples, "pass_at_k": pass_k,
               "temperature": args.temperature}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
