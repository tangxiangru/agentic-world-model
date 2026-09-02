#!/usr/bin/env python3
"""Build the SFT jsonl: {prompt, completion} rendered exactly as the grader renders.

Sources (both derived from the GSM8K TRAIN split only; the test split is never read
except by ../contamination_check.py):
  - openai/gsm8k train, original human solutions
  - nvidia/OpenMathInstruct-2 train_1M, rows with problem_source in
    {gsm8k, augmented_gsm8k} (Llama-3.1-405B solutions, answer-verified by NVIDIA,
    seeded from GSM8K train problems)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import fmt  # noqa: E402

OMI2_GLOB = (
    "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/"
    "train_1M-*.parquet"
)
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def clean_omi2_solution(sol: str, answer: str) -> str | None:
    """Strip the \\boxed marker so the target carries exactly one answer marker."""
    n = len(BOXED.findall(sol))
    if n != 1:
        return None
    sol = BOXED.sub(r"\1", sol)
    if "\\boxed" in sol or "####" in sol or "ANSWER:" in sol:
        return None
    return sol.strip()


def is_plain_integer(a: str) -> bool:
    a = a.strip().replace(",", "")
    return bool(re.fullmatch(r"-?\d+", a))


def load_gsm8k_train(keep_calc_annotations: bool):
    import datasets

    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for rec in ds:
        parts = rec["answer"].split("####")
        target = parts[-1].strip()
        reasoning = "####".join(parts[:-1]).strip()
        if not keep_calc_annotations:
            reasoning = re.sub(r"<<[^>]*>>", "", reasoning)
        out.append((rec["question"].strip(), reasoning, target))
    return out


def load_omi2(max_rows: int, rng: random.Random, pattern: str = OMI2_GLOB):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(pattern))
    assert files, "OpenMathInstruct-2 parquet files not found"
    rows, seen = [], set()
    for path in files:
        df = pq.read_table(path).to_pandas()
        df = df[df["problem_source"].isin(["gsm8k", "augmented_gsm8k"])]
        for problem, sol, ans in zip(
            df["problem"], df["generated_solution"], df["expected_answer"]
        ):
            if not is_plain_integer(ans):
                continue
            body = clean_omi2_solution(sol, ans)
            if body is None or len(body) < 30 or len(body) > 3000:
                continue
            key = (problem.strip(), body[:160])
            if key in seen:
                continue
            seen.add(key)
            rows.append((problem.strip(), body, ans.strip().replace(",", "")))
        if len(rows) >= max_rows * 3:
            break
    rng.shuffle(rows)
    return rows[:max_rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--omi2-rows", type=int, default=48000)
    ap.add_argument("--gsm8k-repeats", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-calc-annotations", type=int, default=1)
    ap.add_argument("--omi2-glob", default=OMI2_GLOB)
    args = ap.parse_args()

    fmt.assert_prompt_template_matches()
    rng = random.Random(args.seed)

    records = []
    gsm = load_gsm8k_train(bool(args.keep_calc_annotations))
    for rep in range(args.gsm8k_repeats):
        # tag the copy so the dedup below keeps deliberate repeats of the original data
        records.extend((q, b, a, f"gsm8k#{rep}") for q, b, a in gsm)
    print(f"gsm8k train rows: {len(gsm)} x{args.gsm8k_repeats}")

    omi2 = load_omi2(args.omi2_rows, rng, args.omi2_glob)
    records.extend((q, b, a, "omi2") for q, b, a in omi2)
    print(f"omi2 gsm8k-source rows: {len(omi2)}")

    rng.shuffle(records)

    seen_prompts = set()
    n_fewshot = 0
    with open(args.out, "w") as f:
        for question, body, answer, tag in records:
            fewshot = rng.random() < args.fewshot_frac
            n_fewshot += fewshot
            prompt = fmt.render_prompt(question, fewshot=fewshot)
            completion = fmt.render_target(body, answer)
            assert completion.count(fmt.ANSWER_MARKER) == 1, completion
            assert completion.rstrip("\n").endswith(fmt.STOP_TOKEN)
            key = (tag, question, body[:200])
            if key in seen_prompts:
                continue
            seen_prompts.add(key)
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "question": question,
                        "answer": answer,
                        "fewshot": bool(fewshot),
                    }
                )
                + "\n"
            )
    print(f"wrote {args.out}: {len(seen_prompts)} rows, {n_fewshot} with the 10-shot prefix")


if __name__ == "__main__":
    main()
