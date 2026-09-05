"""Rejection-sampling data: sample k chains per problem from a checkpoint, keep
the ones whose graded answer matches gold.

Grading uses the same end-anchored numeric rule as the harness (render.is_correct),
so a kept chain is one that would have scored a point.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import EOT, is_correct, render_prompt  # noqa: E402


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n", type=int, default=4, help="samples per problem")
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.problems)]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [render_prompt(r["question"]) for r in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=1536,
        max_num_seqs=768,
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    # No per-request seed and no string stop condition: both cost throughput,
    # and <end_of_turn> (106) is already in the checkpoint's eos_token_id.
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=0.95,
        top_k=64,
        max_tokens=args.max_tokens,
    )
    outs = llm.generate(prompts, sp)

    stats = collections.Counter()
    n_solved = 0
    with open(args.out, "w") as f:
        for row, prompt, out in zip(rows, prompts, outs):
            kept, seen = 0, set()
            solved = False
            for cand in out.outputs:
                text = cand.text.strip()
                stats["sampled"] += 1
                if not is_correct(text, row["gold"]):
                    stats["wrong"] += 1
                    continue
                solved = True
                if text.count("ANSWER:") != 1:
                    stats["marker_count"] += 1
                    continue
                key = norm_text(text)
                if key in seen:
                    stats["dup"] += 1
                    continue
                seen.add(key)
                if kept >= args.keep_per_problem:
                    stats["over_cap"] += 1
                    continue
                kept += 1
                stats["kept"] += 1
                stats[f"kept_{row['src']}"] += 1
                f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "completion": text + EOT,
                            "answer": row["gold"],
                            "src": "rft:" + row["src"],
                        }
                    )
                    + "\n"
                )
            n_solved += int(solved)

    summary = {
        "problems": len(rows),
        "problems_with_at_least_one_correct": n_solved,
        "pass_at_k": round(n_solved / max(len(rows), 1), 4),
        "sample_accuracy": round(
            (stats["sampled"] - stats["wrong"]) / max(stats["sampled"], 1), 4
        ),
        **dict(stats),
    }
    print(json.dumps(summary, indent=2))
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
