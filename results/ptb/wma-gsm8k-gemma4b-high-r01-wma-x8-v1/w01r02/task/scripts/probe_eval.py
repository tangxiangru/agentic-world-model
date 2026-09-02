#!/usr/bin/env python3
"""Fast offline eval / sampler on the held-out gsm8k-TRAIN probe set.

Uses the same prompt string the grader renders (fmt.render_prompt with the
grader's own 10-shot system message) and the same answer extraction
(fmt.grade), so a number from here is comparable in kind to evaluate.py -
but it is measured on probe300, never on the benchmark test split.

Also doubles as the RFT sampler: --n > 1 and --temperature > 0 writes every
sample with its correctness flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/probe300.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--fewshot", type=int, default=1, help="1 = use the grader's 10-shot system message")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--save-samples", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit > 0:
        rows = rows[: args.limit]
    system = open("data/fewshot_system.txt").read() if args.fewshot else None
    prompts = [fmt.render_prompt(r["question"], system) for r in rows]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=0,
    )
    outs = llm.generate(prompts, sp)

    n_correct = 0
    n_capped = 0
    recs = []
    for r, o in zip(rows, outs):
        texts = [c.text for c in o.outputs]
        ok = [fmt.grade(t, r["gold"]) for t in texts]
        n_correct += int(ok[0])
        n_capped += int(o.outputs[0].finish_reason == "length")
        rec = {"id": r["id"], "gold": r["gold"], "correct_first": ok[0], "pass_any": any(ok),
               "n_correct": sum(ok)}
        if args.save_samples:
            rec["question"] = r["question"]
            rec["samples"] = [{"text": t, "ok": k} for t, k in zip(texts, ok)]
        recs.append(rec)

    summary = {
        "model": args.model,
        "n": len(rows),
        "greedy_accuracy": n_correct / len(rows),
        "pass_at_n": sum(r["pass_any"] for r in recs) / len(rows),
        "cap_hit_rate": n_capped / len(rows),
        "temperature": args.temperature,
        "n_samples": args.n,
        "fewshot": bool(args.fewshot),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "items": recs}, f)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
