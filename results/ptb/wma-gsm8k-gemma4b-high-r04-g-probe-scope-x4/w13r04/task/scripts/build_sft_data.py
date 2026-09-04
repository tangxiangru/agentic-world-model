#!/usr/bin/env python3
"""Build the SFT file for the gsm8k post-train.

Source: nvidia/OpenMathInstruct-2 (train_1M split), gsm8k-family rows only.
Every emitted row is (a) rendered with the grader's own gemma3 template,
(b) terminated with <end_of_turn>, and (c) re-scored with inspect_ai's own
match(numeric=True) so a row that would not score cannot enter the file.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import END, fewshot_prefix, render_prompt, render_target  # noqa: E402

from inspect_ai.scorer._common import match_str  # noqa: E402

OMI2 = sorted(
    glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/"
        "469216e3f46f4dacf476b382e192485ea51a143e/data/train_1M-*.parquet"
    )
)
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMERIC_ANS = re.compile(r"^-?\d+(\.\d+)?$")


def clean_solution(sol: str, ans: str) -> str | None:
    n_boxed = len(BOXED.findall(sol))
    if n_boxed != 1:
        return None
    sol = BOXED.sub(r"\1", sol)
    sol = sol.replace("\\dfrac", "\\frac")
    if "ANSWER:" in sol or "####" in sol:
        return None
    sol = sol.strip()
    if not sol:
        return None
    return f"{sol}\n\nANSWER: {ans}"


def scores_ok(target_body: str, ans: str) -> bool:
    """Would the grader read `ans` out of this completion?"""
    _, matched = match_str(value=target_body, target=ans, location="end", numeric=True)
    return matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--max-target-chars", type=int, default=3000)
    ap.add_argument("--sources", default="gsm8k,augmented_gsm8k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keep_sources = set(args.sources.split(","))
    rng = random.Random(args.seed)

    per_problem: dict[str, int] = {}
    seen_target: set[int] = set()
    rows: list[dict] = []
    stats = {"read": 0, "src": 0, "nonnumeric": 0, "boxed": 0, "toolong": 0,
             "unscored": 0, "dupe_problem": 0, "dupe_target": 0, "kept": 0}

    for f in OMI2:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000):
            for r in batch.to_pylist():
                stats["read"] += 1
                if r["problem_source"] not in keep_sources:
                    continue
                stats["src"] += 1
                ans = (r["expected_answer"] or "").strip()
                if not NUMERIC_ANS.match(ans):
                    stats["nonnumeric"] += 1
                    continue
                prob = r["problem"].strip()
                if per_problem.get(prob, 0) >= args.max_per_problem:
                    stats["dupe_problem"] += 1
                    continue
                body = clean_solution(r["generated_solution"] or "", ans)
                if body is None:
                    stats["boxed"] += 1
                    continue
                if len(body) > args.max_target_chars:
                    stats["toolong"] += 1
                    continue
                if not scores_ok(body, ans):
                    stats["unscored"] += 1
                    continue
                h = hash(body)
                if h in seen_target:
                    stats["dupe_target"] += 1
                    continue
                seen_target.add(h)
                per_problem[prob] = per_problem.get(prob, 0) + 1
                rows.append({"question": prob, "body": body, "answer": ans,
                             "source": r["problem_source"]})
                stats["kept"] += 1

    rng.shuffle(rows)
    rows = rows[: args.n]

    fs = fewshot_prefix()
    n_fs = int(len(rows) * args.fewshot_frac)
    for i, r in enumerate(rows):
        r["use_fewshot"] = i < n_fs
    rng.shuffle(rows)  # so the long fewshot rows are spread over the schedule
    with open(args.out, "w") as fh:
        for i, r in enumerate(rows):
            system = fs if r["use_fewshot"] else None
            rec = {
                "prompt": render_prompt(r["question"], system=system),
                "completion": render_target(r["body"]),
                "answer": r["answer"],
                "source": r["source"],
                "fewshot": bool(system),
            }
            assert rec["completion"].rstrip("\n").endswith(END)
            assert rec["completion"].count("ANSWER: ") == 1
            fh.write(json.dumps(rec) + "\n")

    stats["written"] = len(rows)
    stats["fewshot_rows"] = n_fs
    print(json.dumps(stats, indent=1))

    with open(args.out + ".questions.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"text": r["question"] + "\n" + r["body"]}) + "\n")


if __name__ == "__main__":
    main()
