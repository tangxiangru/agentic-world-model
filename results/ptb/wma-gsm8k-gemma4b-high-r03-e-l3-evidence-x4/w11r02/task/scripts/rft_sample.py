"""Rejection-sampling data: draw k chains per question, keep the ones that land.

Reads questions with gold answers, samples from a fine-tuned checkpoint with
the grader's own prompt, and keeps chains whose completion the grader's own
scorer marks correct. Output rows have the same shape build_data.py writes
({question, completion, answer, source}) so train_sft.py can read them directly.
`completion` carries the trailing <end_of_turn> exactly as the trainer needs it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True,
                    help="jsonl with question + answer (gold integer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--n-shot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-tag", default="rft")
    args = ap.parse_args()

    from inspect_ai.scorer._common import match_str
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [fmt.render_prompt(r["question"], args.n_shot, tok) for r in rows]

    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len,
              enable_prefix_caching=True, generation_config="vllm", seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, n=args.k,
                        stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept, n_correct, n_total, solved = [], 0, 0, 0
    for r, o in zip(rows, outs):
        gold = str(r["answer"]).replace(",", "").strip()
        good = []
        for c in o.outputs:
            n_total += 1
            text = c.text
            if c.finish_reason == "length":
                continue
            if not match_str(text, gold, location="end", numeric=True)[1]:
                continue
            body = text.strip()
            # the target the trainer will see must satisfy the same invariants
            if body.count("ANSWER: ") != 1 or not body.rstrip().split("\n")[-1].startswith("ANSWER: "):
                continue
            n_correct += 1
            good.append(body)
        if not good:
            continue
        solved += 1
        seen, uniq = set(), []
        for g in good:
            k = norm_text(g)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(g)
        rng.shuffle(uniq)
        # prefer the shortest correct chains: they are the ones with the least
        # room for a slip, and they keep the epoch cheap
        uniq.sort(key=len)
        for g in uniq[: args.keep_per_question]:
            kept.append({"question": r["question"], "completion": g + fmt.END_OF_TURN,
                         "answer": gold, "source": args.source_tag})

    stats = {
        "model": args.model,
        "questions": len(rows),
        "k": args.k,
        "temperature": args.temperature,
        "samples": n_total,
        "correct_samples": n_correct,
        "sample_accuracy": n_correct / n_total if n_total else None,
        "questions_solved_at_least_once": solved,
        "coverage": solved / len(rows) if rows else None,
        "rows_kept": len(kept),
    }
    print(json.dumps(stats, indent=2), flush=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
