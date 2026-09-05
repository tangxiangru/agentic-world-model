#!/usr/bin/env python3
"""Build the SFT jsonl for GSM8K post-training of gemma-3-4b-pt.

Sources (all GSM8K *train*-derived or fully synthetic; the gsm8k test split is
never read here):
  * openai/gsm8k  main/train  (7473 human solutions, the exact style of the
    10-shot demonstrations the grader puts in the system message)
  * nvidia/OpenMathInstruct-2 train_1M, rows with problem_source in
    {gsm8k, augmented_gsm8k} (Llama-3.1-405B solutions to GSM8K train problems
    and to augmentations of them)

Output rows: {"prompt": <rendered up to '<start_of_turn>model\\n'>,
              "completion": <target + '<end_of_turn>\\n'>, ...}
so a completion-only SFT loss trains exactly the string the grader samples.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".")
    if not NUM_RE.match(s):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def clean_gsm8k_solution(ans_field: str) -> tuple[str, str] | None:
    """openai/gsm8k answer -> (reasoning without <<calc>> annotations, final answer)."""
    if "####" not in ans_field:
        return None
    body, final = ans_field.rsplit("####", 1)
    final = norm_num(final)
    if final is None:
        return None
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    if not body:
        return None
    return body, final


def clean_omi_solution(sol: str, expected: str) -> tuple[str, str] | None:
    """OpenMathInstruct-2 solution -> (reasoning without \\boxed, final answer).

    The \\boxed{} macro is unwrapped in place so the target carries exactly one
    answer marker ("ANSWER: N", appended later) and no competing one.
    """
    final = norm_num(expected)
    if final is None:
        return None
    boxes = BOXED_RE.findall(sol)
    if not boxes:
        return None
    if norm_num(boxes[-1]) != final:
        return None
    body = BOXED_RE.sub(lambda m: m.group(1), sol).strip()
    if "\\boxed" in body or "####" in body:
        return None
    return body, final


def load_gsm8k_train():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        c = clean_gsm8k_solution(r["answer"])
        if c is None:
            continue
        out.append({"question": r["question"].strip(), "reasoning": c[0], "answer": c[1],
                    "src": "gsm8k_train"})
    return out


def load_omi(max_per_problem: int):
    import pandas as pd

    files = sorted(glob.glob(OMI_GLOB))
    assert files, "OpenMathInstruct-2 shards not downloaded"
    by_problem: dict[str, list] = {}
    for f in files:
        df = pd.read_parquet(f, columns=["problem", "generated_solution", "expected_answer",
                                         "problem_source"])
        df = df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])]
        for problem, sol, exp in zip(df["problem"], df["generated_solution"],
                                     df["expected_answer"]):
            c = clean_omi_solution(sol, exp)
            if c is None:
                continue
            by_problem.setdefault(problem.strip(), []).append(c)
    out = []
    for problem, sols in by_problem.items():
        # shortest solutions first: the long ones in OMI-2 are the rambling ones
        sols.sort(key=lambda c: len(c[0]))
        for body, final in sols[:max_per_problem]:
            out.append({"question": problem, "reasoning": body, "answer": final,
                        "src": "omi2_gsm8k"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi", type=int, default=45000)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.25)
    ap.add_argument("--fewshot-k-max", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm = load_gsm8k_train()
    print(f"gsm8k train usable: {len(gsm)}", flush=True)
    omi = load_omi(args.max_per_problem)
    print(f"omi2 gsm8k usable: {len(omi)}", flush=True)

    omi = [r for r in omi if len(r["reasoning"]) <= args.max_chars]
    rng.shuffle(omi)
    omi = omi[: args.n_omi]

    rows = omi + gsm * args.gsm8k_repeat
    rng.shuffle(rows)

    # the few-shot demonstration pool: original gsm8k train solutions, the exact
    # shape inspect_evals puts in the system message
    pool = gsm

    n_written = 0
    with open(args.out, "w") as f:
        for r in rows:
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(2, args.fewshot_k_max)
                shots = rng.sample(pool, k)
                system = "\n\n".join(
                    fmt.fewshot_block(s["question"], s["reasoning"], s["answer"])
                    for s in shots
                )
            prompt = fmt.render(r["question"], system=system, target=None)
            completion = fmt.target_text(r["reasoning"], r["answer"]) + fmt.STOP_TOKEN + "\n"
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                "src": r["src"],
                "fewshot": 0 if system is None else 1,
            }) + "\n")
            n_written += 1
    print(f"wrote {n_written} rows -> {args.out}")


if __name__ == "__main__":
    main()
