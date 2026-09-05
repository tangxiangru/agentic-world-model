#!/usr/bin/env python3
"""Build SFT data for gsm8k, rendered byte-for-byte with the grader's own template.

Sources: nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k rows only).
Output: jsonl with {prompt, completion} where `prompt` is the fully rendered
gemma3.jinja prefix up to '<start_of_turn>model\n' and `completion` is the
target text ending in the grading stop token '<end_of_turn>'.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer

TASK = "/home/ben/task"
BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
OMI = ("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
       "469216e3f46f4dacf476b382e192485ea51a143e/data")

# copied verbatim from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def strip_boxed(sol: str) -> str:
    return BOXED.sub(r"\1", sol)


def norm_answer(a: str) -> str | None:
    """Keep only answers the grader can score: plain numbers."""
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    if a.startswith("\\text{") and a.endswith("}"):
        a = a[6:-1].strip()
    try:
        v = float(a)
    except ValueError:
        return None
    if v == int(v) and abs(v) < 1e12:
        return str(int(v))
    return a


def render(tok, system: str | None, question: str, solution: str) -> tuple[str, str]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)})
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    completion = solution + "<end_of_turn>"
    return prompt, completion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{TASK}/data/sft_v1.jsonl")
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-file", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    skip = set()
    if args.skip_file:
        for line in open(args.skip_file):
            rr = json.loads(line)
            pp = rr["prompt"].split("$ANSWER is the answer to the problem.\n\n", 1)[1]
            skip.add(hashlib.md5(pp.split("\n\nRemember to put your answer on its own line", 1)[0].strip().encode()).hexdigest())
    print(f"skipping {len(skip)} problems", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open(f"{TASK}/templates/gemma3.jinja").read()
    sysmsg = open(f"{TASK}/data/fewshot_system.txt").read()

    files = sorted(glob.glob(f"{OMI}/train-*.parquet"))
    print(f"{len(files)} parquet shards")

    per_problem: dict[str, int] = {}
    seen_sol: set[str] = set()
    rows: list[dict] = []
    n_read = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=20000,
                                     columns=["problem", "generated_solution",
                                              "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                n_read += 1
                if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
                    continue
                ans = norm_answer(r["expected_answer"])
                if ans is None:
                    continue
                q = r["problem"].strip()
                k = hashlib.md5(q.encode()).hexdigest()
                if k in skip or per_problem.get(k, 0) >= args.max_per_problem:
                    continue
                sol = strip_boxed(r["generated_solution"]).strip()
                if "ANSWER:" in sol or "\\boxed" in sol or "####" in sol:
                    continue
                if len(sol) < 30 or len(sol) > 4000:
                    continue
                # the grader reads the LAST numeric token: the answer line must be last.
                target = f"{sol}\n\nANSWER: {ans}"
                sk = hashlib.md5(target.encode()).hexdigest()
                if sk in seen_sol:
                    continue
                seen_sol.add(sk)
                per_problem[k] = per_problem.get(k, 0) + 1
                rows.append({"question": q, "target": target, "answer": ans,
                             "source": r["problem_source"]})
        print(f"  {os.path.basename(fp)}: read={n_read} kept={len(rows)}", flush=True)
        if len(rows) >= args.n * 1.3:
            break

    rng.shuffle(rows)
    rows = rows[: args.n]
    print(f"selected {len(rows)}")

    n_fs = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            use_fs = rng.random() < args.fewshot_frac
            n_fs += use_fs
            prompt, completion = render(tok, sysmsg if use_fs else None,
                                        r["question"], r["target"])
            assert completion.endswith("<end_of_turn>")
            assert completion.count("ANSWER: ") == 1, completion[-200:]
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "answer": r["answer"], "source": r["source"],
                                "fewshot": use_fs}) + "\n")
    print(f"wrote {args.out}  ({n_fs} with fewshot prefix)")


if __name__ == "__main__":
    main()
