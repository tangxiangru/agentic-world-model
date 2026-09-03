#!/usr/bin/env python3
"""Build the round-1 SFT corpus.

Sources (all GSM8K *train* derived or synthetic augmentations of it; the GSM8K
test split is never touched):
  * openai/gsm8k  main + socratic, split=train  (7473 problems)
  * nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}

Every row is written as {"prompt": <rendered gemma3 prompt>, "completion":
<solution + <end_of_turn>>, "target": same as completion} so the preflight
checks can read it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fmt import EOT, fmt_number, render_prompt, render_target  # noqa: E402

from datasets import load_dataset  # noqa: E402

CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def clean_gsm8k(answer: str) -> tuple[str, str] | None:
    if "####" not in answer:
        return None
    body, final = answer.rsplit("####", 1)
    num = fmt_number(final)
    if num is None:
        return None
    body = CALC.sub("", body).strip()
    body = re.sub(r"[ \t]+", " ", body)
    if not body:
        return None
    return body, num


def clean_omi(sol: str, expected: str) -> tuple[str, str] | None:
    num = fmt_number(expected)
    if num is None:
        return None
    s = sol.strip()
    # drop the trailing "The final answer is $\boxed{...}$." sentence
    lines = s.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and "\\boxed" in lines[-1]:
        lines.pop()
    s = "\n".join(lines).strip()
    if not s:
        return None
    if "\\boxed" in s:
        s = BOXED.sub(r"\1", s)
    return s, num


def solution_text(body: str, num: str) -> str:
    return f"{body}\n\nANSWER: {num}"


def ok(sol: str) -> bool:
    if sol.count("ANSWER:") != 1:
        return False
    if EOT in sol or "<start_of_turn>" in sol:
        return False
    if len(sol) > 4000 or len(sol) < 20:
        return False
    return True


def pair_key(prompt: str, completion: str) -> str:
    """Identify a (question, solution) pair regardless of prefix / rendering."""
    head = prompt.split("\n\nRemember to put your answer on its own line")[0]
    q = head.split("\n\n")[-1]
    return (q.strip()[:160] + "||" + completion.strip()[:160])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi", type=int, default=60000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=None,
                    help="jsonl already built; its (prompt, completion) pairs are skipped")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    exclude: set[str] = set()
    if args.exclude:
        for line in open(args.exclude):
            r = json.loads(line)
            exclude.add(pair_key(r["prompt"], r["completion"]))
        print(f"exclusion set: {len(exclude)} pairs", flush=True)

    # ---- the exact 10-shot system message the grader builds -----------------
    from build_fewshot import fewshot_system_message

    system = fewshot_system_message()

    rows: list[tuple[str, str, str]] = []  # (question, solution, source)

    # ---- openai/gsm8k train -------------------------------------------------
    for name in ("main", "socratic"):
        ds = load_dataset("openai/gsm8k", name, split="train")
        for r in ds:
            c = clean_gsm8k(r["answer"])
            if c is None:
                continue
            body, num = c
            sol = solution_text(body, num)
            if ok(sol):
                rows.append((r["question"], sol, f"gsm8k-{name}"))
    n_gsm = len(rows)
    if args.gsm_repeat > 1:
        rows = rows * args.gsm_repeat
    print(f"gsm8k train rows: {n_gsm} (x{args.gsm_repeat})", flush=True)

    # ---- OpenMathInstruct-2 -------------------------------------------------
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    keep_src = {"gsm8k", "augmented_gsm8k"}
    per_problem: dict[str, int] = {}
    omi: list[tuple[str, str, str]] = []
    for r in ds:
        if r["problem_source"] not in keep_src:
            continue
        p = r["problem"]
        if per_problem.get(p, 0) >= args.max_per_problem:
            continue
        c = clean_omi(r["generated_solution"], r["expected_answer"])
        if c is None:
            continue
        body, num = c
        sol = solution_text(body, num)
        if not ok(sol):
            continue
        if exclude and pair_key(render_prompt(p), render_target(sol)) in exclude:
            continue
        per_problem[p] = per_problem.get(p, 0) + 1
        omi.append((p, sol, r["problem_source"]))
    print(f"OpenMathInstruct-2 gsm8k-family rows available: {len(omi)}", flush=True)
    rng.shuffle(omi)
    rows.extend(omi[: args.n_omi])

    rng.shuffle(rows)

    n_fs = 0
    with open(args.out, "w") as f:
        for q, sol, src in rows:
            use_fs = rng.random() < args.fewshot_frac
            n_fs += int(use_fs)
            prompt = render_prompt(q, system if use_fs else None)
            completion = render_target(sol)
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                "target": completion,
                "source": src,
                "fewshot": use_fs,
            }) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({n_fs} with the 10-shot prefix)")


if __name__ == "__main__":
    main()
