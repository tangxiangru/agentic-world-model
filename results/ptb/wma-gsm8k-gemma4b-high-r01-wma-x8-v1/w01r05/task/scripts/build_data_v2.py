#!/usr/bin/env python3
"""Second-round corpus: fresh GSM8K-derived rows disjoint from data/sft_v1.jsonl.

Same extraction and formatting rules as scripts/build_data.py (imported, not
copied, so the two corpora cannot drift), but over 20 OpenMathInstruct-2 shards
instead of 6, and with every (problem, body) pair already used in round one
removed. That makes a second training stage an epoch over *new* problems rather
than a second pass over the same ones.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import (  # noqa: E402
    MATH_PROMPT_TEMPLATE,
    STOP_TOKEN,
    clean_body,
    extract_boxed,
    gsm8k_train_shots,
    load_shot_pool,
    norm_num,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=140000)
    ap.add_argument("--exclude", default=os.path.join(ROOT, "data/sft_v1.jsonl"))
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--max-shots", type=int, default=4)
    ap.add_argument("--max-sol-chars", type=int, default=2200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/sft_v2.jsonl"))
    args = ap.parse_args()

    used_problem: dict[str, int] = defaultdict(int)
    used_pair: set[tuple[str, str]] = set()
    with open(args.exclude) as f:
        for line in f:
            r = json.loads(line)
            p = r["prompt"]
            i = p.find("Solve the following math problem step by step.")
            prob = p[i:].split("\n\n", 1)[1].rsplit("\n\nRemember to put your answer", 1)[0].strip()
            body = r["completion"].rsplit("\n\nANSWER: ", 1)[0]
            used_problem[prob] += 1
            used_pair.add((prob, body))
    print(f"excluding {len(used_pair)} pairs over {len(used_problem)} problems from round one")

    rng = random.Random(args.seed)
    load_shot_pool()

    shards = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/*.parquet"
        )
    )
    print(f"{len(shards)} shards")

    per_problem = defaultdict(int, used_problem)
    seen: set[tuple[str, str]] = set(used_pair)
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
            if gold is None or len(sol) > args.max_sol_chars:
                stats["drop_basic"] += 1
                continue
            prob = problem.strip()
            if per_problem[prob] >= args.max_per_problem:
                stats["drop_problem_cap"] += 1
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
            if not body or (prob, body) in seen:
                stats["drop_dupe_or_empty"] += 1
                continue
            seen.add((prob, body))
            per_problem[prob] += 1
            rows.append({"problem": prob, "body": body, "answer": gold, "source": src})
        print(f"  {os.path.basename(sh)}: kept {len(rows)}", flush=True)
        if len(rows) >= args.n * 1.15:
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
            f.write(
                json.dumps(
                    {
                        "prompt": user,
                        "completion": f"{r['body']}\n\nANSWER: {r['answer']}{STOP_TOKEN}",
                        "answer": r["answer"],
                        "source": r["source"],
                        "n_shots": k,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} rows to {args.out} ({n_fs} few-shot)")


if __name__ == "__main__":
    main()
