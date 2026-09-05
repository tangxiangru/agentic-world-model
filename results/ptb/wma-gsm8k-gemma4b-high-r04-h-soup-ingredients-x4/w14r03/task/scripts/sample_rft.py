#!/usr/bin/env python3
"""Rejection-sampling: draw k solutions per training question from a checkpoint,
keep the ones whose final ANSWER matches the reference.

Questions come from the training corpus only (OpenMathInstruct-2 gsm8k subset +
openai/gsm8k train). The GSM8K test split is never read here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from build_data import norm_answer, tail_number  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question + completion (reference)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-questions", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.questions)][: a.n_questions]
    refs = []
    for r in rows:
        ans = tail_number(r["completion"].replace(common.STOP_TOKEN, ""))
        if ans is not None:
            refs.append({"question": r["question"], "answer": ans, "source": r["source"]})
    print(f"{len(refs)} questions with a readable reference answer", flush=True)

    tok = common.get_tokenizer(a.model)
    prompts = [common.render_prompt(tok, r["question"]) for r in refs]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=a.model,
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=a.k,
        temperature=a.temperature,
        top_p=a.top_p,
        max_tokens=a.max_tokens,
        stop_token_ids=[1, 106],
        seed=0,
    )
    outs = llm.generate(prompts, sp)

    kept, n_correct, n_total = [], 0, 0
    per_q_correct = []
    for r, o in zip(refs, outs):
        seen, good = set(), 0
        for c in o.outputs:
            n_total += 1
            text = c.text.strip()
            if not text:
                continue
            if text.count(common.ANSWER_MARKER) != 1:
                continue
            got = tail_number(text)
            if got is None or got != r["answer"]:
                continue
            good += 1
            n_correct += 1
            if text in seen or len(seen) >= a.keep_per_question:
                continue
            seen.add(text)
            kept.append(
                {
                    "question": r["question"],
                    "completion": text + common.STOP_TOKEN,
                    "source": "rft:" + r["source"],
                }
            )
        per_q_correct.append(good)

    stats = {
        "questions": len(refs),
        "samples": n_total,
        "correct_samples": n_correct,
        "pass_at_1_estimate": n_correct / max(1, n_total),
        "questions_with_no_correct": sum(1 for g in per_q_correct if g == 0),
        "kept_rows": len(kept),
    }
    print("RFT STATS:", json.dumps(stats), flush=True)
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
