#!/usr/bin/env python3
"""Build a second, disjoint SFT shard from OpenMathInstruct-2 train_2M:
gsm8k-derived rows whose (problem, solution) pair is NOT already in an existing
jsonl file. Same formatting rules as scripts/prepare_data.py."""
import argparse, collections, json, random

from prepare_data import CALC_RE, MATH_PROMPT_TEMPLATE, make_row, norm_int, strip_boxed  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--max-per-problem", type=int, default=4)
ap.add_argument("--max-rows", type=int, default=10 ** 9)
ap.add_argument("--seed", type=int, default=2)
a = ap.parse_args()

rng = random.Random(a.seed)
seen = set()
per_problem = collections.Counter()
for path in a.exclude:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            seen.add(r["text"])
            per_problem[r["prompt"]] += 1
print(f"excluding {len(seen)} existing rows over {len(per_problem)} prompts")

from datasets import load_dataset
d = load_dataset("nvidia/OpenMathInstruct-2", split="train_2M")
rows = []
for r in d:
    if r["problem_source"] not in ("gsm8k", "augmented_gsm8k"):
        continue
    ans = norm_int(r["expected_answer"])
    if ans is None:
        continue
    sol = strip_boxed(r["generated_solution"]).strip()
    if not sol or len(sol) > 6000:
        continue
    row = make_row(r["problem"], sol, ans, "omi2v2_" + r["problem_source"])
    if row["text"] in seen:
        continue
    if per_problem[row["prompt"]] >= a.max_per_problem:
        continue
    seen.add(row["text"])
    per_problem[row["prompt"]] += 1
    row["completion"] += "<end_of_turn>"
    rows.append(row)

rng.shuffle(rows)
rows = rows[: a.max_rows]
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} new rows -> {a.out}")
print(collections.Counter(r["source"] for r in rows))
