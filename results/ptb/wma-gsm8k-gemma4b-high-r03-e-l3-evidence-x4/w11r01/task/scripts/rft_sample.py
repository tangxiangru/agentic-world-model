#!/usr/bin/env python3
"""Rejection-sampling data: let the SFT model solve training questions, keep the
chains whose final answer is right.

Prompts are rendered with the grader's own chat template so the samples are
on-policy for the distribution the model is actually graded in, and each kept
chain already ends in "ANSWER: <n><end_of_turn>" because that is what the model
was trained to emit -- verified here rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response '
    'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    "answer to the problem.\n\n{prompt}\n\nRemember to put your answer on its own line "
    'at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    "answer to the problem, and you do not need to use a \\boxed command.\n\nReasoning:"
)
END_OF_TURN = "<end_of_turn>"
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> str | None:
    m = NUM.findall(text)
    if not m:
        return None
    v = m[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
    except ValueError:
        return None
    return str(int(f)) if f.is_integer() else str(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-questions", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-question", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    template = open(TEMPLATE).read()
    rows = [json.loads(l) for l in open(args.questions)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_questions]

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=r["question"])}],
            chat_template=template, tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1536, dtype="bfloat16", seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept: dict[int, list[str]] = defaultdict(list)
    n_gen = n_right = 0
    solved = 0
    for i, o in enumerate(outs):
        gold = rows[i]["answer"]
        good = []
        for c in o.outputs:
            n_gen += 1
            txt = c.text.strip()
            if final_number(txt) != gold or "ANSWER:" not in txt:
                continue
            if txt.count("ANSWER:") != 1 or not txt.split("ANSWER:")[0].strip():
                continue
            n_right += 1
            good.append(txt)
        if good:
            solved += 1
        # prefer the shortest correct chains: they are the least likely to ramble
        good.sort(key=len)
        seen = set()
        for g in good:
            if g in seen:
                continue
            seen.add(g)
            kept[i].append(g)
            if len(kept[i]) >= args.keep_per_question:
                break

    out_rows = []
    for i, chains in kept.items():
        for c in chains:
            out_rows.append({"question": rows[i]["question"],
                             "target": c + END_OF_TURN,
                             "answer": rows[i]["answer"]})
    rng.shuffle(out_rows)
    with open(args.out, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", "") + ".decon.jsonl", "w") as fh:
        for r in out_rows:
            fh.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")

    stats = {"questions": len(rows), "k": args.k, "generations": n_gen,
             "correct_generations": n_right, "pass_at_k_questions": solved / len(rows),
             "sample_accuracy": n_right / n_gen, "rows_written": len(out_rows)}
    print(json.dumps(stats, indent=2))
    json.dump(stats, open(args.out.replace(".jsonl", "") + ".stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
