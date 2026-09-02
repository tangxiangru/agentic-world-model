#!/usr/bin/env python3
"""Build the SFT corpus.

Sources (all public, none derived from the GSM8K *test* split):
  * nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k,
    math, augmented_math}.  gsm8k/math rows are the original *training*
    problems of those benchmarks with model-written solutions; the augmented_*
    rows are new problems synthesised from those training problems.
  * openai/gsm8k train split -- the gold reference solutions, reformatted.

Every target is shaped for the grader: chain of thought, then a final line
"ANSWER: <n>", then <end_of_turn>.  \\boxed{} is unwrapped so no second answer
marker survives (pitfall: double_answer_format).
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

OMI2_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"

NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")
BOXED = re.compile(r"\\boxed\{")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    if a.endswith(".0"):
        a = a[:-2]
    return a if NUMERIC.match(a) else None


def unwrap_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    while True:
        m = BOXED.search(text)
        if not m:
            return text
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        inner = text[m.end(): i - 1]
        text = text[: m.start()] + inner + text[i:]


def clean_solution(sol: str, answer: str) -> str | None:
    sol = unwrap_boxed(sol).strip()
    if "ANSWER:" in sol or "\\boxed" in sol:
        return None
    # strip a trailing "The final answer is 17." style line: the ANSWER line replaces it
    sol = re.sub(r"\n*(The (final )?answer is[^\n]*)$", "", sol).strip()
    if not sol:
        return None
    return f"{sol}\n\nANSWER: {answer}"


def gsm8k_gold(row) -> str | None:
    """Reformat an openai/gsm8k train solution: drop <<calc>> spans and ####."""
    ans_field = row["answer"]
    if "####" not in ans_field:
        return None
    body, final = ans_field.rsplit("####", 1)
    ans = norm_answer(final)
    if ans is None:
        return None
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    if not body:
        return None
    return f"{body}\n\nANSWER: {ans}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-aug-gsm8k", type=int, default=90000)
    ap.add_argument("--n-gsm8k", type=int, default=20000)
    ap.add_argument("--n-math", type=int, default=15000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=None,
                    help="jsonl whose 'question' values must not appear in the output")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=32)
    ap.add_argument("--no-gold", action="store_true", help="skip the openai/gsm8k gold rows")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    exclude = set()
    if args.exclude:
        for line in open(args.exclude):
            exclude.add(json.loads(line)["question"].strip())
        print(f"excluding {len(exclude)} already-used questions", flush=True)
    import pandas as pd
    import pyarrow.parquet as pq

    want = {
        "augmented_gsm8k": args.n_aug_gsm8k,
        "gsm8k": args.n_gsm8k,
        "math": args.n_math // 2,
        "augmented_math": args.n_math - args.n_math // 2,
    }
    got: dict[str, int] = {k: 0 for k in want}
    per_problem: dict[str, int] = {}
    seen_sol: set[int] = set()
    rows: list[dict] = []

    shards = [p for p in sorted(glob.glob(OMI2_GLOB))
              if args.shard_start <= int(os.path.basename(p).split("-")[1]) < args.shard_end]
    print("shards:", [os.path.basename(p) for p in shards], flush=True)
    for path in shards:
        if all(got[k] >= want[k] for k in want):
            break
        df = pq.read_table(path).to_pandas()
        df = df.sample(frac=1.0, random_state=args.seed)  # break shard ordering
        for src, prob, sol, ans in zip(
            df["problem_source"], df["problem"], df["generated_solution"], df["expected_answer"]
        ):
            if src not in want or got[src] >= want[src]:
                continue
            if len(sol) > args.max_sol_chars:
                continue
            a = norm_answer(ans)
            if a is None:
                continue
            key = prob.strip()
            if key in exclude:
                continue
            if per_problem.get(key, 0) >= args.max_per_problem:
                continue
            target = clean_solution(sol, a)
            if target is None:
                continue
            h = hash((key, target))
            if h in seen_sol:
                continue
            seen_sol.add(h)
            per_problem[key] = per_problem.get(key, 0) + 1
            got[src] += 1
            rows.append({"question": key, "solution": target, "answer": a, "source": f"omi2:{src}"})
        print(f"{os.path.basename(path)}: {got}", flush=True)

    # openai/gsm8k train gold solutions -- the style the grader's own few-shots use
    from datasets import load_dataset

    ds = [] if args.no_gold else load_dataset("openai/gsm8k", "main")["train"]
    n_gold = 0
    for row in ds:
        t = gsm8k_gold(row)
        if t is None:
            continue
        rows.append(
            {
                "question": row["question"].strip(),
                "solution": t,
                "answer": t.rsplit("ANSWER: ", 1)[1],
                "source": "gsm8k_train_gold",
            }
        )
        n_gold += 1
    print("gsm8k gold:", n_gold, flush=True)

    # few-shot prefixes: a minority of rows carry k in-context examples so the
    # model learns to keep its own solution style when the grader prepends 10.
    pool = [r for r in rows if r["source"] == "gsm8k_train_gold"]
    if not pool:
        from datasets import load_dataset as _ld
        pool = []
        for row in _ld("openai/gsm8k", "main")["train"].select(range(2000)):
            t = gsm8k_gold(row)
            if t:
                pool.append({"question": row["question"].strip(), "solution": t})
    rng.shuffle(rows)
    n_fs = int(len(rows) * args.fewshot_frac)
    for i, r in enumerate(rows):
        if i < n_fs:
            k = rng.choice([2, 4, 10])
            shots = rng.sample(pool, k)
            blocks = []
            for s in shots:
                body, ans = s["solution"].rsplit("\n\nANSWER: ", 1)
                blocks.append(fmt.fewshot_block(s["question"], body, ans))
            r["system"] = "\n\n".join(blocks)
        else:
            r["system"] = None

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            prompt, target = fmt.build_example(r["question"], r["solution"], system=r["system"])
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": target,
                        "question": r["question"],
                        "answer": r["answer"],
                        "source": r["source"],
                        "n_shot": 0 if r["system"] is None else r["system"].count("\n\nANSWER: "),
                    }
                )
                + "\n"
            )
    print("wrote", len(rows), "rows to", args.out)


if __name__ == "__main__":
    main()
