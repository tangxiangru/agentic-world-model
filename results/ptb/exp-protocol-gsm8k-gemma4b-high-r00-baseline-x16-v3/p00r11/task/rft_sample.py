#!/usr/bin/env python3
"""Rejection-sampling data generation from a fine-tuned checkpoint.

Samples k solutions per training question with vLLM, keeps the ones whose final
'ANSWER: N' line matches the gold answer, and writes them in the same schema
build_sft.py uses.  Questions come from the GSM8K *train* split and from
OpenMathInstruct-2's gsm8k-family problems; no test item is ever read.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from build_sft import MATH_PROMPT_TEMPLATE, norm_answer

ANSWER_RE = re.compile(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)\s*$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--n-questions", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-keep-per-question", type=int, default=2)
    ap.add_argument("--exclude-questions", default=None,
                    help="jsonl whose 'question' field lists problems to skip")
    ap.add_argument("--gsm8k-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    skip = set()
    if args.exclude_questions:
        with open(args.exclude_questions) as f:
            skip = {json.loads(l)["question"].strip() for l in f}

    items = []  # (question, gold)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for rec in gsm:
        a = norm_answer(rec["answer"].rpartition("####")[2])
        if a is not None:
            items.append((rec["question"].strip(), a))
    n_gsm = len(items)

    if not args.gsm8k_only and len(items) < args.n_questions:
        omi = load_dataset("nvidia/OpenMathInstruct-2", split="train")
        omi = omi.filter(
            lambda x: x["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=16
        )
        idx = list(range(len(omi)))
        rng.shuffle(idx)
        seen = {q for q, _ in items}
        for i in idx:
            if len(items) >= args.n_questions:
                break
            rec = omi[i]
            q = rec["problem"].strip()
            if q in seen or q in skip:
                continue
            a = norm_answer(rec["expected_answer"])
            if a is None:
                continue
            seen.add(q)
            items.append((q, a))
    items = items[: args.n_questions]
    print(f"[rft] {len(items)} questions ({n_gsm} from gsm8k train)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open("templates/gemma3.jinja").read()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            chat_template=template, tokenize=False, add_generation_prompt=True,
        )
        for q, _ in items
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=2048,
              dtype="bfloat16", enforce_eager=False)
    # <end_of_turn> (106) is the terminator the grading template stops on;
    # <eos> (1) is the tokenizer's. Passing them explicitly: vLLM's offline
    # entrypoint does not pick them up from generation_config here, and without
    # them the model re-emits the ANSWER line until max_tokens.
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, seed=args.seed,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    stats = Counter()
    per_q_correct = Counter()
    rows = []
    raw_dump = open(args.out + ".raw", "w")   # crash insurance: generations first
    for (q, gold), out in zip(items, outs):
        raw_dump.write(json.dumps({"q": q, "gold": gold,
                                   "gen": [c.text for c in out.outputs]}) + "\n")
        correct_texts: list[str] = []
        for cand in out.outputs:
            text = cand.text.strip()
            m = ANSWER_RE.search(text)
            stats["samples"] += 1
            if m is None:
                stats["no_answer_line"] += 1
                continue
            if norm_answer(m.group(1)) != gold:
                stats["wrong"] += 1
                continue
            stats["correct"] += 1
            per_q_correct[q] += 1
            if text in correct_texts:
                stats["dup"] += 1
                continue
            correct_texts.append(text)
        # a question the model already solves every time teaches little: keep one
        # sample of it, but up to --max-keep-per-question of the ones it only
        # sometimes gets right
        budget = 1 if per_q_correct[q] == args.k else args.max_keep_per_question
        kept = sorted(correct_texts, key=len)[:budget]
        stats["kept"] += len(kept)
        for text in kept:
            rows.append({
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=q),
                "completion": text + "<end_of_turn>",
                "question": q,
                "answer": gold,
                "system": None,
                "source": "rft_self",
                "n_correct_of_k": per_q_correct[q],
            })

    raw_dump.close()
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    solved = sum(1 for q, _ in items if per_q_correct[q] > 0)
    summary = {
        "n_questions": len(items),
        "k": args.k,
        "rows_written": len(rows),
        "questions_with_a_correct_sample": solved,
        "pass_at_k": solved / max(1, len(items)),
        "sample_accuracy": stats["correct"] / max(1, stats["samples"]),
        "counters": dict(stats),
        "hardness_histogram": dict(Counter(per_q_correct[q] for q, _ in items)),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.stats_out:
        json.dump(summary, open(args.stats_out, "w"), indent=2)


if __name__ == "__main__":
    main()
