#!/usr/bin/env python3
"""Tokenise the SFT corpus with the grader's own chat template.

Prompt/target split, completion-only labels, and a few-shot prefix on a
configurable share of rows so the model is not surprised by the grader's
10-shot system message.

Writes a .pt with a list of {"input_ids": [...], "n_prompt": int} plus a
sidecar .stats.json with the length distribution (pitfall seq_len_truncation).
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
from transformers import AutoTokenizer

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def render_prompt(tok, system: str | None, question: str) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_shot_pool(rows: list[dict], n: int, seed: int) -> list[str]:
    """Few-shot blocks in the grader's own sample_to_fewshot shape, built from
    gsm8k_train rows only (reference reasoning, no LaTeX)."""
    pool = [r for r in rows if r["source"] == "gsm8k_train"]
    random.Random(seed).shuffle(pool)
    out = []
    for r in pool[:n]:
        body = r["completion"]
        body = body.replace("<end_of_turn>", "")
        reasoning, _, ans = body.rpartition("\n\nANSWER: ")
        out.append(f"{r['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {ans.strip()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--p-eval-fewshot", type=float, default=0.10,
                    help="share of rows carrying the grader's exact 10-shot system message")
    ap.add_argument("--p-rand-fewshot", type=float, default=0.20,
                    help="share of rows carrying a random 2-5 shot system message")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(os.path.join(TASK_DIR, "templates", "gemma3.jinja")) as f:
        tok.chat_template = f.read()

    rows = [json.loads(l) for l in open(args.data)]
    eval_system = open(os.path.join(TASK_DIR, "data", "eval_fewshot_system.txt")).read()
    shot_pool = build_shot_pool(rows, 2000, args.seed)

    rng = random.Random(args.seed)
    examples = []
    n_trunc = 0
    lens = []
    for r in rows:
        u = rng.random()
        if u < args.p_eval_fewshot:
            system = eval_system
        elif u < args.p_eval_fewshot + args.p_rand_fewshot:
            k = rng.randint(2, 5)
            system = "\n\n".join(rng.sample(shot_pool, k))
        else:
            system = None
        prompt = render_prompt(tok, system, r["question"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        ids = p_ids + c_ids
        lens.append(len(ids))
        if len(ids) > args.max_seq_len:
            n_trunc += 1
            continue
        examples.append({"input_ids": ids, "n_prompt": len(p_ids)})

    lens.sort()
    stats = {
        "n_rows_in": len(rows),
        "n_rows_kept": len(examples),
        "n_dropped_too_long": n_trunc,
        "share_dropped": n_trunc / max(1, len(rows)),
        "max_seq_len": args.max_seq_len,
        "len_p50": lens[len(lens) // 2],
        "len_p95": lens[int(len(lens) * 0.95)],
        "len_max": lens[-1],
        "total_tokens": sum(len(e["input_ids"]) for e in examples),
        "total_target_tokens": sum(len(e["input_ids"]) - e["n_prompt"] for e in examples),
    }
    torch.save(examples, args.out)
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    # one rendered example, for the record
    ex = examples[0]
    with open(args.out + ".example.txt", "w") as f:
        f.write("=== PROMPT ===\n")
        f.write(tok.decode(ex["input_ids"][: ex["n_prompt"]]))
        f.write("\n=== TARGET (loss on these) ===\n")
        f.write(tok.decode(ex["input_ids"][ex["n_prompt"]:]))


if __name__ == "__main__":
    main()
