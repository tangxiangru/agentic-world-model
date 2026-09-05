"""Rejection-sampling data generation: sample k solutions per training question
from a checkpoint, keep the ones whose graded answer matches gold, and write
them out in the same {question, completion, gold, src} schema build_data.py
uses (completion already terminated with <end_of_turn>).

Questions come only from GSM8K's *train* split and from OpenMathInstruct-2 /
MetaMathQA problems, which are themselves derived from that train split. The
benchmark test set is never read here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_utils as P  # noqa: E402

STOP = "<end_of_turn>"


def norm_body(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question/gold")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--fewshot", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [P.eval_prompt(tok, r["question"], fewshot=bool(args.fewshot)) for r in rows]

    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=3072, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=0)
    outs = llm.generate(prompts, sp)

    kept, seen = [], defaultdict(set)
    n_correct_any = 0
    per_q_correct = []
    for r, o in zip(rows, outs):
        gold = str(r["gold"])
        good = []
        for c in o.outputs:
            if c.finish_reason != "stop":
                continue
            body = c.text.strip()
            if body.count("ANSWER: ") != 1:
                continue
            if not P.grade(body, gold):
                continue
            # the grader reads the last number: make sure it is the gold one
            if not re.search(r"ANSWER:\s*-?[\d,\.]+\s*$", body):
                continue
            good.append(body)
        per_q_correct.append(len(good))
        n_correct_any += bool(good)
        good.sort(key=len)  # prefer the shortest correct chains
        for body in good[: args.max_per_question]:
            key = norm_body(body)
            if key in seen[r["question"]]:
                continue
            seen[r["question"]].add(key)
            kept.append({"question": r["question"], "completion": body + STOP,
                         "gold": gold, "src": "rft"})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {"model": args.model_path, "questions": len(rows), "k": args.k,
             "temperature": args.temperature,
             "questions_with_a_correct_sample": n_correct_any,
             "solve_rate_any": n_correct_any / len(rows),
             "mean_correct_per_question": sum(per_q_correct) / len(rows),
             "unsolved": sum(1 for c in per_q_correct if c == 0),
             "rows_kept": len(kept), "out": args.out}
    print(json.dumps(stats, indent=1))
    if args.stats_out:
        json.dump(stats, open(args.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
