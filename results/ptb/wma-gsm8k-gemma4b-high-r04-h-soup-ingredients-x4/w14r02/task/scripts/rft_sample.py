#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k solutions per training question
from a checkpoint, keep the ones whose 'ANSWER: n' matches the gold answer.

Questions come from the GSM8K *train* split and from the OpenMathInstruct-2
gsm8k-family pool. No test item is ever read.
"""
from __future__ import annotations

import argparse
import glob
import zlib
import json
import random
import re

import pyarrow.parquet as pq

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def num(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def extract(text: str):
    m = list(re.finditer(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", text))
    return num(m[-1].group(1)) if m else None


def load_questions(n_train_repeat: int, n_omi: int, seed: int, exclude: set[str]):
    from datasets import load_dataset

    out = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        gold = num(r["answer"].rsplit("####", 1)[1])
        if gold is not None:
            out.extend([(r["question"].strip(), gold)] * n_train_repeat)

    if n_omi:
        pool = []
        for f in sorted(glob.glob(OMI2)):
            d = pq.read_table(
                f, columns=["problem", "expected_answer", "problem_source"]
            ).to_pydict()
            for q, a, s in zip(d["problem"], d["expected_answer"], d["problem_source"]):
                if s not in ("gsm8k", "augmented_gsm8k"):
                    continue
                g = num(a)
                if g is None:
                    continue
                q = q.strip()
                if q in exclude:
                    continue
                pool.append((q, g))
        seen, uniq = set(), []
        for q, g in pool:
            if q in seen:
                continue
            seen.add(q)
            uniq.append((q, g))
        random.Random(seed).shuffle(uniq)
        out.extend(uniq[:n_omi])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--n-train-repeat", type=int, default=1)
    ap.add_argument("--n-omi", type=int, default=0)
    ap.add_argument("--keep-per-q", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--exclude-file", default=None, help="jsonl whose 'question' field is skipped")
    ap.add_argument("--use-fewshot", action="store_true",
                    help="generate under the grader's exact 10-shot system block")
    args = ap.parse_args()

    exclude = set()
    if args.exclude_file:
        for line in open(args.exclude_file):
            q = json.loads(line).get("question")
            if q:
                exclude.add(q.strip())
        print(f"excluding {len(exclude)} questions already in the SFT mix")

    qs = load_questions(args.n_train_repeat, args.n_omi, args.seed, exclude)
    print(f"{len(qs)} question slots")

    from vllm import LLM, SamplingParams

    tmpl = open("/home/ben/task/templates/gemma3.jinja").read()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=2816 if args.use_fewshot else 1536,
        enable_prefix_caching=True,
    )
    # <end_of_turn> (106) must be given explicitly: LLM.chat does not pick up the
    # second entry of generation_config's eos_token_id list, so without this every
    # sample runs to max_tokens repeating its own answer line and finish_reason is
    # never "stop". That silently rejected 99.9% of a first 80-minute sampling run.
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_token_ids=[1, 106],
    )
    head = []
    if args.use_fewshot:
        from add_fewshot import eval_system_message

        head = [{"role": "system", "content": eval_system_message()}]
        print(f"generating under the grader's 10-shot block ({len(head[0]['content'])} chars)")
    convs = [head + [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=q)}] for q, _ in qs]
    outs = llm.chat(convs, sp, chat_template=tmpl)

    kept = 0
    solved = 0
    n_out = 0
    with open(args.out, "w") as fh:
        for (q, gold), o in zip(qs, outs):
            cands = []
            for c in o.outputs:
                if c.finish_reason != "stop":
                    continue
                t = c.text.strip()
                v = extract(t)
                if v is None or abs(v - gold) > 1e-6:
                    continue
                # exactly one marker, and nothing after it
                m = list(re.finditer(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", t))
                if len(m) != 1 or t[m[0].end():].strip() != "":
                    continue
                cands.append(t)
            n_out += len(o.outputs)
            if cands:
                solved += 1
            # keep up to keep_per_q distinct correct solutions, chosen at random
            # (picking the shortest biases towards chains that got lucky)
            cands = sorted(set(cands))
            random.Random(zlib.crc32(q.encode())).shuffle(cands)
            cands = cands[: args.keep_per_q]
            for t in cands:
                kept += 1
                fh.write(
                    json.dumps(
                        {
                            "prompt": PROMPT_TEMPLATE.format(prompt=q),
                            "completion": t,
                            "question": q,
                            "text": q + "\n" + t,
                        }
                    )
                    + "\n"
                )
    print(
        f"samples {n_out}  questions with >=1 correct {solved}/{len(qs)} "
        f"({solved / len(qs):.3f})  rows kept {kept} -> {args.out}"
    )


if __name__ == "__main__":
    main()
