#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K from OpenMathInstruct-2 + the GSM8K train split.

Output: jsonl rows {"system": str|null, "question": str, "target": str}
`target` always ends with the answer line "ANSWER: <n>" and nothing after it,
because the grader (inspect_ai match(numeric=True, location="end")) reads the
LAST number of the completion.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"

# byte-for-byte the template inspect_evals/gsm8k uses for the user turn
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

INT_RE = re.compile(r"^-?\d{1,12}$")
BOXED_RE = re.compile(r"\\boxed\{")
CALC_RE = re.compile(r"<<[^>]*>>")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{X} with X (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])
        j = m.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        out.append(text[m.end() : j - 1])
        i = j
    return "".join(out)


def tidy(sol: str) -> str:
    sol = strip_boxed(sol)
    sol = CALC_RE.sub("", sol)
    # drop a trailing "The answer is ..." / "So the answer is ..." fragment left dangling
    sol = sol.rstrip()
    return sol


def build_target(sol: str, ans: str) -> str | None:
    sol = tidy(sol)
    if not sol:
        return None
    return f"{sol}\n\nANSWER: {ans}"


def fewshot_system() -> str:
    """Reproduce the exact 10-shot system message inspect_evals/gsm8k builds."""
    from inspect_ai.dataset import hf_dataset
    from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot

    shots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=10,
    )
    return "\n\n".join(sample_to_fewshot(s) for s in shots)


def load_gsm8k_train():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for r in ds:
        q = r["question"].strip()
        a = r["answer"]
        body, _, ans = a.partition("####")
        ans = ans.strip().replace(",", "")
        if not INT_RE.match(ans):
            continue
        body = CALC_RE.sub("", body).strip()
        rows.append((q, f"{body}\n\nANSWER: {ans}", ans))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-gsm", type=int, default=120000)
    ap.add_argument("--n-math", type=int, default=15000)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(glob.glob(OMI2_GLOB))
    assert files, "OpenMathInstruct-2 parquet files not found"

    by_problem: dict[str, list[str]] = defaultdict(list)
    math_pool: list[tuple[str, str, str]] = []
    stats = Counter()

    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000):
            d = batch.to_pydict()
            for prob, sol, ans, src in zip(
                d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]
            ):
                ans = (ans or "").strip().replace(",", "")
                if not INT_RE.match(ans):
                    stats["drop_nonint"] += 1
                    continue
                if src in ("gsm8k", "augmented_gsm8k"):
                    if len(by_problem[prob]) >= args.max_per_problem:
                        stats["drop_cap"] += 1
                        continue
                    t = build_target(sol, ans)
                    if t is None:
                        stats["drop_empty"] += 1
                        continue
                    by_problem[prob].append(t)
                    stats["keep_gsm"] += 1
                elif src in ("math", "augmented_math"):
                    if len(math_pool) < args.n_math * 4:
                        t = build_target(sol, ans)
                        if t is not None:
                            math_pool.append((prob, t, ans))
                    stats["keep_math_pool"] += 1

    gsm_rows = [(p, t, None) for p, ts in by_problem.items() for t in ts]
    rng.shuffle(gsm_rows)
    gsm_rows = gsm_rows[: args.n_gsm]

    rng.shuffle(math_pool)
    math_rows = [(p, t, None) for p, t, _ in math_pool[: args.n_math]]

    orig = load_gsm8k_train()
    orig_rows = [(q, t, None) for q, t, _ in orig] * args.gsm8k_repeat

    rows = gsm_rows + math_rows + orig_rows
    rng.shuffle(rows)

    sysmsg = fewshot_system()
    n_fs = int(len(rows) * args.fewshot_frac)

    with open(args.out, "w") as fh:
        for i, (q, t, _) in enumerate(rows):
            fh.write(
                json.dumps({"system": sysmsg if i < n_fs else None, "question": q, "target": t})
                + "\n"
            )

    print("stats", dict(stats))
    print(
        f"unique gsm problems {len(by_problem)}  gsm rows {len(gsm_rows)}  "
        f"math rows {len(math_rows)}  orig rows {len(orig_rows)}  total {len(rows)}  fewshot rows {n_fs}"
    )
    print("system message chars:", len(sysmsg))


if __name__ == "__main__":
    main()
