#!/usr/bin/env python3
"""Sample k solutions per problem from a checkpoint with vLLM, keep the ones
whose final ANSWER matches the gold answer, and write an SFT-ready jsonl.

Problem sources are GSM8K *train* questions and OpenMathInstruct-2
gsm8k-derived problems (also train-derived).  The GSM8K test split is never
read here.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fmt import (  # noqa: E402
    MATH_PROMPT_TEMPLATE,
    STOP_TOKEN,
    fewshot_system_message,
    load_tokenizer,
    render_prompt,
)

NUMLIKE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> str | None:
    m = NUMLIKE.findall(text)
    if not m:
        return None
    s = m[-1].replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_problems(args) -> list[dict]:
    probs: list[dict] = []
    if args.gsm8k_train:
        import pyarrow.parquet as pq

        p = (
            "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/"
            "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"
        )
        for r in pq.read_table(p).to_pylist():
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            probs.append({"problem": r["question"].strip(), "answer": gold, "src": "gsm8k_train"})
    if args.from_jsonl:
        seen = set()
        with open(args.from_jsonl) as f:
            for line in f:
                r = json.loads(line)
                if r["problem"] in seen:
                    continue
                seen.add(r["problem"])
                probs.append({"problem": r["problem"], "answer": r["answer"], "src": "omi2"})
    random.Random(args.seed).shuffle(probs)
    if args.max_problems:
        probs = probs[: args.max_problems]
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-train", action="store_true")
    ap.add_argument("--from-jsonl", default=None)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--labels-out", default=None,
                    help="jsonl of {problem, answer, k, n_correct} for every problem sampled")
    args = ap.parse_args()

    tok = load_tokenizer()
    probs = load_problems(args)
    print(f"{len(probs)} problems", flush=True)

    sysmsg = fewshot_system_message() if args.fewshot else None
    prompts = [render_prompt(tok, p["problem"], system=sysmsg) for p in probs]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=3072 if args.fewshot else 1024,
        enable_prefix_caching=True,
        dtype="bfloat16",
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[tok.convert_tokens_to_ids(STOP_TOKEN), tok.eos_token_id],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    stats = collections.Counter()
    per_problem_correct = []
    labels = []
    rows = []
    for p, o in zip(probs, outs):
        kept, seen_texts = 0, set()
        n_correct = 0
        for c in o.outputs:
            txt = c.text.strip()
            stats["samples"] += 1
            got = last_number(txt)
            if got is None or got != p["answer"]:
                stats["wrong"] += 1
                continue
            n_correct += 1
            if "ANSWER:" not in txt or txt.count("ANSWER:") != 1:
                stats["bad_marker"] += 1
                continue
            if txt in seen_texts or kept >= args.keep_per_problem:
                stats["dup_or_capped"] += 1
                continue
            seen_texts.add(txt)
            kept += 1
            rows.append(
                {
                    "problem": p["problem"],
                    "completion": txt + STOP_TOKEN,
                    "answer": p["answer"],
                    "source": "rft:" + p["src"],
                    "prompt": MATH_PROMPT_TEMPLATE.format(prompt=p["problem"]),
                }
            )
            stats["kept"] += 1
        per_problem_correct.append(n_correct)
        labels.append({"problem": p["problem"], "answer": p["answer"],
                       "src": p["src"], "k": args.k, "n_correct": n_correct})
        if n_correct == 0:
            stats["problems_all_wrong"] += 1

    random.Random(args.seed + 1).shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "n_problems": len(probs),
        "k": args.k,
        "temperature": args.temperature,
        "pass_at_1": round(sum(per_problem_correct) / max(1, args.k * len(probs)), 4),
        "pass_at_k": round(
            sum(1 for c in per_problem_correct if c > 0) / max(1, len(probs)), 4
        ),
        "counts": dict(stats),
        "rows_written": len(rows),
        "out": args.out,
    }
    if args.labels_out:
        with open(args.labels_out, "w") as f:
            for r in labels:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote labels", args.labels_out)
    print(json.dumps(summary, indent=2))
    if args.stats_out:
        json.dump(summary, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
