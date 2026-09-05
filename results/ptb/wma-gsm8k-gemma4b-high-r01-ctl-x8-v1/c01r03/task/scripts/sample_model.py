#!/usr/bin/env python3
"""Sample completions from a checkpoint with vLLM, under the grader's prompt.

Two uses:
  --mode dev   : one greedy completion per question of our own dev holdout
                 (gsm8k TRAIN split, never trained on) -> a cheap accuracy
                 signal that does not touch the benchmark test set.
  --mode rft   : k sampled completions per question, for rejection-sampling
                 fine-tuning.

Prompts are rendered with scripts/fmt.py, i.e. templates/gemma3.jinja, and by
default carry the grader's exact 10-shot prefix so the samples are on-policy
for the real evaluation. vLLM prefix-caches that shared prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmt import render_prompt_fast  # noqa: E402
from eval_format import build_system_message, build_user_message  # noqa: E402

from inspect_ai.scorer._common import match_str  # noqa: E402

# The grader is match(numeric=True) -> match_str(location="end"). Call the very
# same function rather than reimplementing it, so "correct" here means exactly
# what it means in evaluate.py.


def grade(completion: str, gold: str) -> tuple[str, bool]:
    return match_str(
        value=completion,
        target=str(gold),
        location="end",
        ignore_case=True,
        numeric=True,
    )


def norm_gold(g: str) -> str:
    return str(g).replace(",", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question/gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["dev", "rft"], default="dev")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fewshot", type=int, default=1, help="1 = grader's 10-shot prefix")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = []
    with open(args.questions) as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
    if args.limit:
        rows = rows[: args.limit]

    system = build_system_message() if args.fewshot else None
    prompts = [render_prompt_fast(system, build_user_message(r["question"])) for r in rows]

    if args.mode == "dev":
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, n=1,
                            stop_token_ids=[1, 106])
    else:
        sp = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            n=args.k,
            stop_token_ids=[1, 106],
        )

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        dtype="bfloat16",
    )
    outs = llm.generate(prompts, sp)

    n_correct = 0
    n_items = 0
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            gold = norm_gold(r["gold"])
            samples = []
            for c in o.outputs:
                txt = c.text
                pred, ok = grade(txt, gold)
                samples.append(
                    {
                        "text": txt,
                        "pred": pred,
                        "correct": bool(ok),
                        "finish": c.finish_reason,
                    }
                )
            n_items += 1
            if samples[0]["correct"]:
                n_correct += 1
            f.write(
                json.dumps(
                    {
                        "id": r.get("id"),
                        "question": r["question"],
                        "gold": r["gold"],
                        "gold_norm": gold,
                        "samples": samples,
                    }
                )
                + "\n"
            )

    print(json.dumps({"n": n_items, "first_sample_acc": n_correct / max(1, n_items)}))
    if args.mode == "rft":
        # pass@k for reference
        pass_k = sum(
            1
            for line in open(args.out)
            if any(s["correct"] for s in json.loads(line)["samples"])
        )
        print(json.dumps({"pass_at_k": pass_k / max(1, n_items), "k": args.k}))


if __name__ == "__main__":
    main()
