"""Offline vLLM eval on a held-out slice of the gsm8k TRAIN split.

Reproduces the grader end to end -- the same 10-shot system block
(inspect_evals.gsm8k, fewshot_seed=42), the same chat template, the same
end-anchored numeric match -- but on items that were excluded from training,
so it gives a signal that never touches the test split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import EOT, fewshot_system, is_correct, render_prompt  # noqa: E402


def build_fewshot(n: int, seed: int = 42) -> str:
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample

    shots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=seed,
        limit=n,
    )
    return fewshot_system(
        [(s.input, s.metadata["reasoning"], s.target) for s in shots]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", default="/home/ben/task/data/dev_train.jsonl")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--fewshot", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dev)][: args.limit]
    system = build_fewshot(args.fewshot) if args.fewshot else None
    prompts = [render_prompt(r["question"], system=system) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=8192,
        enforce_eager=False,
        dtype="bfloat16",
    )
    if args.greedy:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=[EOT])
    else:
        sp = SamplingParams(
            temperature=1.0, top_p=0.95, top_k=64, max_tokens=args.max_tokens,
            stop=[EOT], seed=0,
        )
    outs = llm.generate(prompts, sp)

    n_ok = 0
    n_fmt = 0
    recs = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        ok = is_correct(text, r["gold"])
        fmt_bad = "ANSWER:" not in text
        n_ok += ok
        n_fmt += fmt_bad
        recs.append({"id": r["id"], "gold": r["gold"], "correct": bool(ok),
                     "no_answer_line": fmt_bad, "n_tokens": len(o.outputs[0].token_ids),
                     "completion": text})
    summary = {
        "model": args.model,
        "n": len(rows),
        "accuracy": round(n_ok / max(len(rows), 1), 4),
        "no_answer_line": n_fmt,
        "greedy": args.greedy,
        "fewshot": args.fewshot,
        "mean_tokens": round(sum(r["n_tokens"] for r in recs) / max(len(recs), 1), 1),
    }
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "items": recs}, f, indent=2)


if __name__ == "__main__":
    main()
