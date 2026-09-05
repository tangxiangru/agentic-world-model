#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K.

Source: nvidia/OpenMathInstruct-2 (rev 469216e3f46f4dacf476b382e192485ea51a143e),
the gsm8k and augmented_gsm8k slices only. Those are Llama-3.1-405B solutions to
GSM8K *train* problems and to problems augmented from GSM8K *train*; the GSM8K
test split is not part of that dataset. Every row is still run through the
harness contamination checker afterwards.

Target shape is dictated by the grader, not by taste:
  - inspect_evals/gsm8k asks for a final line 'ANSWER: $ANSWER'
  - the scorer is match(location='end', numeric=True): it reads the LAST numeric
    word of the completion, so the answer line must really be last
  - templates/gemma3.jinja terminates an assistant turn with '<end_of_turn>'

Writes JSONL with {prompt, completion, answer, source, n_shots}.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_sft import STOP_TOKEN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUMERIC = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$")
BOXED = re.compile(r"\\boxed\{")


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUMERIC.match(s):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def extract_boxed(sol: str) -> tuple[str, str] | None:
    """Return (solution with \\boxed{X} flattened to X, X) or None."""
    m = BOXED.search(sol)
    if m is None:
        return None
    i = m.end()  # first char inside the brace
    depth = 1
    j = i
    while j < len(sol) and depth:
        if sol[j] == "{":
            depth += 1
        elif sol[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    if depth:
        return None
    inner = sol[i:j]
    flat = sol[: m.start()] + inner + sol[j + 1 :]
    return flat, inner


def clean_body(body: str) -> str:
    body = body.replace("\\[", "").replace("\\]", "")
    body = body.replace("\\(", "").replace("\\)", "")
    body = re.sub(r"\\text\{([^{}]*)\}", r"\1", body)
    body = re.sub(r"\\times", "*", body)
    body = re.sub(r"\\cdot", "*", body)
    body = re.sub(r"\\div", "/", body)
    body = re.sub(r"\\%", "%", body)
    body = body.replace("$", "")
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def gsm8k_train_shots(rng: random.Random, k: int) -> str:
    """Few-shot prefix in exactly the shape inspect_evals builds, from TRAIN items."""
    picks = rng.sample(_SHOT_POOL, k)
    return "\n\n".join(picks)


_SHOT_POOL: list[str] = []


def load_shot_pool() -> None:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        reasoning, _, ans = r["answer"].rpartition("####")
        reasoning = reasoning.strip()
        ans = ans.strip()
        if len(reasoning) > 700:
            continue
        _SHOT_POOL.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {ans}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120000, help="rows to emit")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--max-shots", type=int, default=4)
    ap.add_argument("--max-sol-chars", type=int, default=2200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "data/sft_v1.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    load_shot_pool()
    print(f"shot pool: {len(_SHOT_POOL)} gsm8k-train exemplars")

    shards = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/*.parquet"
        )
    )
    print(f"{len(shards)} shards")

    per_problem: dict[str, int] = defaultdict(int)
    seen_sol: set[int] = set()
    rows: list[dict] = []
    stats = defaultdict(int)

    for sh in shards:
        df = pq.read_table(sh).to_pandas()
        df = df[df["problem_source"].isin(["gsm8k", "augmented_gsm8k"])]
        for problem, sol, exp, src in zip(
            df["problem"], df["generated_solution"], df["expected_answer"], df["problem_source"]
        ):
            stats["seen"] += 1
            gold = norm_num(exp)
            if gold is None:
                stats["drop_nonnumeric_gold"] += 1
                continue
            if per_problem[problem] >= args.max_per_problem:
                stats["drop_problem_cap"] += 1
                continue
            if len(sol) > args.max_sol_chars:
                stats["drop_long"] += 1
                continue
            got = extract_boxed(sol)
            if got is None:
                stats["drop_no_boxed"] += 1
                continue
            flat, inner = got
            if norm_num(inner) != gold:
                stats["drop_boxed_mismatch"] += 1
                continue
            body = clean_body(flat)
            if not body:
                stats["drop_empty"] += 1
                continue
            h = hash((problem, body))
            if h in seen_sol:
                stats["drop_dupe"] += 1
                continue
            seen_sol.add(h)
            per_problem[problem] += 1
            rows.append(
                {"problem": problem.strip(), "body": body, "answer": gold, "source": src}
            )
        print(f"  {os.path.basename(sh)}: kept {len(rows)}", flush=True)
        if len(rows) >= args.n * 1.3:
            break

    rng.shuffle(rows)
    rows = rows[: args.n]
    print("stats:", dict(stats))

    n_fs = 0
    with open(args.out, "w") as f:
        for r in rows:
            user = MATH_PROMPT_TEMPLATE.replace("{prompt}", r["problem"])
            k = 0
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, args.max_shots)
                user = gsm8k_train_shots(rng, k) + "\n\n" + user
                n_fs += 1
            # the stop token is part of the supervised target, not a rendering
            # detail: the grader's vLLM stops on <end_of_turn> (token 106)
            completion = f"{r['body']}\n\nANSWER: {r['answer']}{STOP_TOKEN}"
            f.write(
                json.dumps(
                    {
                        "prompt": user,
                        "completion": completion,
                        "answer": r["answer"],
                        "source": r["source"],
                        "n_shots": k,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} rows to {args.out} ({n_fs} with a few-shot prefix)")


if __name__ == "__main__":
    main()
