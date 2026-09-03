#!/usr/bin/env python3
"""Rejection sampling: draw k solutions per problem from a trained checkpoint,
keep the ones whose graded answer matches the known answer.

Correctness is judged with the grader's own function
(inspect_ai.scorer._match.match_str, numeric, location="end") so a "correct"
sample here is correct for the benchmark too.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from inspect_ai.scorer._match import match_str

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

INT_RE = re.compile(r"^-?\d{1,12}$")


def collect_problems(n_max: int, seed: int, offset: int = 0) -> list[tuple[str, str]]:
    """(question, answer) over gsm8k train + OMI2 gsm8k/augmented_gsm8k problems."""
    from datasets import load_dataset

    seen: dict[str, str] = {}
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        if INT_RE.match(ans):
            seen[r["question"].strip()] = ans
    n_gsm = len(seen)

    for f in sorted(glob.glob(OMI2_GLOB)):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000, columns=["problem", "expected_answer", "problem_source"]):
            d = batch.to_pydict()
            for p, a, s in zip(d["problem"], d["expected_answer"], d["problem_source"]):
                if s not in ("gsm8k", "augmented_gsm8k"):
                    continue
                a = (a or "").strip().replace(",", "")
                if INT_RE.match(a):
                    seen.setdefault(p.strip(), a)

    items = list(seen.items())
    random.Random(seed).shuffle(items)
    print(f"problem pool: {len(items)} distinct ({n_gsm} from gsm8k train); offset {offset}", flush=True)
    items = items[offset:]
    return items[:n_max] if n_max else items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=30000)
    ap.add_argument("--problem-offset", type=int, default=0,
                    help="skip the first N problems of the (seed-shuffled) pool, so a later "
                         "round can draw problems an earlier round never saw")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--keep-easy-frac", type=float, default=0.3,
                    help="problems the model already solves k/k are mostly redundant; "
                         "keep this fraction of them, one solution each")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(TEMPLATE).read()

    problems = collect_problems(args.n_problems, args.seed, args.problem_offset)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q, _ in problems
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=1024,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<end_of_turn>"],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    stats = Counter()
    passk = Counter()
    rng = random.Random(args.seed)
    kept = 0
    side = open(args.out + ".passrate.jsonl", "w")
    with open(args.out, "w") as fh:
        for (q, gold), o in zip(problems, outs):
            good = []
            for c in o.outputs:
                txt = c.text.strip()
                stats["samples"] += 1
                if txt.count("ANSWER:") != 1:
                    stats["bad_marker"] += 1
                    continue
                _, ok = match_str(value=txt, target=gold, location="end", numeric=True)
                if ok:
                    good.append(txt)
            passk[f"{len(good)}/{args.k}"] += 1
            side.write(json.dumps({"question": q, "answer": gold, "n_correct": len(good), "k": args.k}) + "\n")
            if not good:
                stats["problem_unsolved"] += 1
                continue
            if len(good) == args.k:
                # already solved every time: little learning signal, keep a sample of them
                # only so the model does not drift away from what it gets right
                if rng.random() > args.keep_easy_frac:
                    stats["skip_easy"] += 1
                    continue
                cap = 1
                stats["kept_easy"] += 1
            else:
                cap = args.max_per_problem
                stats["kept_uncertain"] += 1
            # prefer the shortest correct solutions: less rambling, fewer chances to slip
            good = sorted(set(good), key=len)[:cap]
            for t in good:
                fh.write(json.dumps({"system": None, "question": q, "target": t}) + "\n")
                kept += 1

    side.close()
    print("sample stats", dict(stats), flush=True)
    print("solved-count histogram", dict(sorted(passk.items())), flush=True)
    print(f"kept {kept} rows over {len(problems)} problems -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
