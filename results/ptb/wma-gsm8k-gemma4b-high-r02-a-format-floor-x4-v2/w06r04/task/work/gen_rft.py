"""Rejection sampling: draw k solutions per problem from a checkpoint, keep the
ones that reach the reference answer, and write them back in training format.

No external model is called - the samples come from the checkpoint being
improved. Problems come from the same two sources as the SFT file (gsm8k
train[0:7273] and OpenMathInstruct-2 gsm8k slices); the reference answer is the
dataset's, so a kept solution is verified, not merely plausible.

Usage:
  python work/gen_rft.py --model work/sft_v1 --problems work/data/rft_problems.jsonl \
      --out work/data/rft_v1.jsonl --k 4 --temperature 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_eval import render, user_message  # noqa: E402

ANS = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.M)
STOP = "<end_of_turn>"


def num(x):
    x = x.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(x)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.30)
    ap.add_argument("--min-shots", type=int, default=3)
    ap.add_argument("--max-shots", type=int, default=10)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--select", choices=["shortest", "median", "longest"], default="median")
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    probs = [json.loads(l) for l in open(args.problems)]
    # Sampling prompts are ZERO-SHOT: shorter to generate, and the few-shot
    # prefix is re-attached (to a random 30% of rows) only when the kept
    # solutions are written out as training rows.
    prompts = [render([{"role": "user", "content": user_message(p["question"])}],
                      add_generation_prompt=True) for p in probs]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=2048, dtype="bfloat16", disable_log_stats=True)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, n=args.k, seed=args.seed)
    outs = llm.generate(prompts, sp)

    from build_data import fewshot_pool  # same pool, same rendering as the SFT file

    pool = fewshot_pool()
    rng = random.Random(args.seed)

    n_any = n_rows = 0
    per_k_solved = [0] * (args.k + 1)
    with open(args.out, "w") as f:
        for p, o in zip(probs, outs):
            gold = num(str(p["gold"]))
            kept, seen = [], set()
            for c in o.outputs:
                t = c.text.strip()
                m = ANS.findall(t)
                if not m or c.finish_reason != "stop":
                    continue
                if num(m[-1]) is None or gold is None or abs(num(m[-1]) - gold) > 1e-6:
                    continue
                if t.count("ANSWER: ") != 1:
                    continue
                key = " ".join(t.split()).lower()
                if key in seen:
                    continue
                seen.add(key)
                kept.append(t)
            per_k_solved[min(len(kept), args.k)] += 1
            if not kept:
                continue
            n_any += 1
            # exp-04 kept the SHORTEST correct chains and mean reasoning length
            # collapsed 213 -> 158 tokens, costing 4.5 pp on the benchmark. Keep the
            # chains closest to this problem's MEDIAN correct-chain length instead, so
            # selection preserves the model's length prior rather than shrinking it.
            if args.select == "median":
                lens = sorted(len(t) for t in kept)
                med = lens[len(lens) // 2]
                kept.sort(key=lambda t: abs(len(t) - med))
            elif args.select == "longest":
                kept.sort(key=len, reverse=True)
            else:
                kept.sort(key=len)
            for t in kept[: args.keep_per_problem]:
                shots = None
                if rng.random() < args.fewshot_frac:
                    kk = rng.randint(args.min_shots, args.max_shots)
                    shots = [s for s in rng.sample(pool, kk + 1) if s[0] != p["question"]][:kk]
                msgs = []
                if shots:
                    msgs.append({"role": "system",
                                 "content": "\n\n".join(b for _, b in shots)})
                msgs.append({"role": "user", "content": user_message(p["question"])})
                f.write(json.dumps({
                    "id": f"rft-{n_rows}:{p['id']}",
                    "source": f"rft:{os.path.basename(args.model.rstrip('/'))}",
                    "question": p["question"],
                    "prompt": render(msgs, add_generation_prompt=True),
                    "completion": t + STOP,
                }) + "\n")
                n_rows += 1

    stats = {"problems": len(probs), "k": args.k, "select": args.select, "solved_at_least_once": n_any,
             "pass_at_k": n_any / len(probs), "rows_written": n_rows,
             "distinct_correct_per_problem_histogram": per_k_solved,
             "temperature": args.temperature, "model": args.model}
    print(json.dumps(stats, indent=2))
    if args.stats:
        json.dump(stats, open(args.stats, "w"), indent=2)


if __name__ == "__main__":
    main()
