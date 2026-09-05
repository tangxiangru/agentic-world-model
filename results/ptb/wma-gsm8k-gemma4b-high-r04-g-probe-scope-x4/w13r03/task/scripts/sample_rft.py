#!/usr/bin/env python3
"""Rejection-sampling data generation: sample k chains per training problem from
a checkpoint, keep the ones whose ANSWER line matches gold.

Problems come from the GSM8K TRAIN split and from OpenMathInstruct-2's
gsm8k / augmented_gsm8k problems (both GSM8K-train derived). The test split is
never read.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANSWER_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    return None


def load_problems(n_omi: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    probs: dict[str, str] = {}
    for r in pq.read_table(sorted(glob.glob(GSM8K_TRAIN))[0]).to_pylist():
        a = norm_num(r["answer"].rsplit("####", 1)[-1])
        if a is not None:
            probs[r["question"]] = a
    n_gsm = len(probs)
    omi: dict[str, str] = {}
    for path in sorted(glob.glob(OMI2)):
        t = pq.read_table(path, columns=["problem", "expected_answer", "problem_source"])
        for p, a, s in zip(
            t.column("problem").to_pylist(),
            t.column("expected_answer").to_pylist(),
            t.column("problem_source").to_pylist(),
        ):
            if s not in ("gsm8k", "augmented_gsm8k") or p in probs or p in omi:
                continue
            a = norm_num(a)
            if a is not None:
                omi[p] = a
    keys = sorted(omi)
    rng.shuffle(keys)
    for k in keys[:n_omi]:
        probs[k] = omi[k]
    print(f"[problems] gsm8k train {n_gsm} + OpenMathInstruct-2 {len(probs) - n_gsm} = {len(probs)}")
    return list(probs.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-problems", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    chat_template = open(TEMPLATE).read()
    problems = load_problems(args.n_omi, args.seed)
    if args.max_problems:
        problems = problems[: args.max_problems]

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        for q, _ in problems
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        seed=args.seed,
        enforce_eager=False,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[1, 106],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    rng = random.Random(args.seed)
    kept = 0
    n_solved = 0
    per_problem_correct = []
    with open(args.out, "w") as f:
        for (q, gold), prompt, out in zip(problems, prompts, outs):
            cands = []
            seen = set()
            for o in out.outputs:
                txt = o.text.strip()
                m = ANSWER_RE.search(txt)
                if not m or norm_num(m.group(1)) != gold:
                    continue
                h = hashlib.md5(txt.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                cands.append(txt)
            per_problem_correct.append(len(cands))
            if not cands:
                continue
            n_solved += 1
            cands.sort(key=len)
            # weight the frontier: a problem the model already gets right on
            # every sample teaches it almost nothing, so keep one chain for
            # those and the full quota for the ones it only sometimes solves
            quota = 1 if len(cands) == args.k else args.max_per_problem
            for txt in cands[:quota]:
                f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "completion": txt + "<end_of_turn>",
                            "question": q,
                            "gold": gold,
                        }
                    )
                    + "\n"
                )
                kept += 1

    n = len(problems)
    print(f"[rft] problems {n}; solved at least once {n_solved} ({n_solved/n:.1%}); rows kept {kept}")
    print(f"[rft] mean correct-of-{args.k} per problem: {sum(per_problem_correct)/n:.2f}")
    print(f"[rft] wrote {args.out}")


if __name__ == "__main__":
    main()
