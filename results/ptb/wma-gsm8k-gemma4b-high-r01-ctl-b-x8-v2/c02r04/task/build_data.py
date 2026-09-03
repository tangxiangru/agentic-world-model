#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Sources
  nvidia/OpenMathInstruct-2 (rev 469216e3, train_1M shard set):
    problem_source in {gsm8k, augmented_gsm8k}  -> the bulk of the corpus
    problem_source in {math, augmented_math}    -> a small numeric-answer tail
  openai/gsm8k (rev 740312ad, main/train)       -> few-shot prefixes only

Target shape (what the grader reads):
    <reasoning prose>

    ANSWER: <number>
followed by the grading template's terminator <end_of_turn>.

Nothing here touches the GSM8K test split.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

OMI2 = sorted(glob.glob(
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"))
GSM8K_TRAIN = glob.glob(
    "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet")[0]

# byte-for-byte the prompt the grader builds (inspect_evals/gsm8k/gsm8k.py L27-35)
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"^-?\d{1,12}(\.\d{1,6})?$")


def unbox(text: str) -> str:
    """Replace every \\boxed{X} with X (brace-balanced)."""
    while True:
        i = text.find("\\boxed{")
        if i < 0:
            return text
        j = i + len("\\boxed{")
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        inner = text[i + len("\\boxed{"): j - 1]
        text = text[:i] + inner + text[j:]


TRAILING_JUNK = re.compile(r"(?:\\\[|\\\]|\$|\s|\.|:|\*)+$")


def clean_solution(sol: str, ans: str) -> str | None:
    s = unbox(sol).strip()
    # drop a dangling final line that is now just the bare answer / empty math
    lines = s.split("\n")
    while lines:
        tail = TRAILING_JUNK.sub("", lines[-1].strip())
        tail_norm = tail.replace("$", "").replace("\\[", "").replace("\\]", "").strip()
        if tail_norm == "" or tail_norm == ans.strip():
            lines.pop()
        else:
            break
    s = "\n".join(lines).rstrip()
    if len(s) < 30:
        return None
    return s


def sample_to_fewshot(q: str, a: str) -> str:
    """inspect_evals/gsm8k sample_to_fewshot, verbatim in shape."""
    reasoning = "####".join(a.split("####")[:-1]).strip()
    target = a.split("####")[-1].strip()
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-gsm", type=int, default=140000)
    ap.add_argument("--n-math", type=int, default=12000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- few-shot pool from the GSM8K TRAIN split -------------------------
    gt = pq.read_table(GSM8K_TRAIN).to_pylist()
    fewshot_pool = [sample_to_fewshot(r["question"], r["answer"]) for r in gt]
    print(f"few-shot pool: {len(fewshot_pool)} gsm8k train items")

    # ---- OpenMathInstruct-2 ----------------------------------------------
    per_problem: dict[str, int] = defaultdict(int)
    gsm_rows, math_rows = [], []
    seen_pairs = set()
    for f in OMI2:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=20000):
            for r in batch.to_pylist():
                src = r["problem_source"]
                ans = (r["expected_answer"] or "").strip()
                if not NUM_RE.match(ans):
                    continue
                is_gsm = src in ("gsm8k", "augmented_gsm8k")
                if not is_gsm and len(math_rows) >= args.n_math * 3:
                    continue
                prob = r["problem"].strip()
                if per_problem[prob] >= args.max_per_problem:
                    continue
                sol = clean_solution(r["generated_solution"], ans)
                if sol is None:
                    continue
                key = (prob, sol[:200])
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                per_problem[prob] += 1
                (gsm_rows if is_gsm else math_rows).append((prob, sol, ans, src))
    print(f"candidates: gsm={len(gsm_rows)} math={len(math_rows)}")

    rng.shuffle(gsm_rows)
    rng.shuffle(math_rows)
    rows = gsm_rows[: args.n_gsm] + math_rows[: args.n_math]
    rng.shuffle(rows)

    n_fs = 0
    with open(args.out, "w") as fh:
        for prob, sol, ans, src in rows:
            messages = []
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 3, 4, 6, 10])
                shots = rng.sample(fewshot_pool, k)
                messages.append({"role": "system", "content": "\n\n".join(shots)})
                n_fs += 1
            messages.append(
                {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=prob)}
            )
            completion = f"{sol}\n\nANSWER: {ans}"
            messages.append({"role": "assistant", "content": completion})
            fh.write(json.dumps({
                "messages": messages,
                "completion": completion,
                "question": prob,
                "answer": ans,
                "source": src,
            }) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({n_fs} with a few-shot prefix)")


if __name__ == "__main__":
    main()
