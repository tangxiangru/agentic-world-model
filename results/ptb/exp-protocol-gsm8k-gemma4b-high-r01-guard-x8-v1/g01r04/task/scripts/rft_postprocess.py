#!/usr/bin/env python3
"""Turn raw rejection-sampling generations into SFT rows.

vLLM's offline generate() does not stop on <end_of_turn> (its stop set comes
from the tokenizer's eos, id 1, not from generation_config's [1, 106]), so a
sample often runs on into a second turn after its answer. Cutting each sample
at its FIRST 'ANSWER: <n>' line is what makes the row match what the grader
would have seen, and keeps continuation text out of the training targets.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

ANS_LINE = re.compile(r"^ANSWER:[ \t]*(-?[\d,]+(?:\.\d+)?)[ \t]*$", re.M)


def norm(v: str) -> str | None:
    v = v.replace(",", "").rstrip(".")
    try:
        f = float(v)
    except (ValueError, OverflowError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return str(int(f)) if f == int(f) else str(f)


def first_answer(text: str):
    """(body_up_to_and_including_the_answer_line, normalised answer) or None."""
    m = ANS_LINE.search(text)
    if not m:
        return None
    return text[: m.end()].strip(), norm(m.group(1))


ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stats-out", default=None)
ap.add_argument("--keep-per-q", type=int, default=2)
ap.add_argument("--max-chars", type=int, default=3000)
ap.add_argument("--drop-if-all-correct", action="store_true",
                help="skip questions the model already solves in every sample")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rows = []
per_src = {}
n_q = n_solved = n_samples = n_correct = n_no_marker = n_ran_on = 0
for line in open(args.raw):
    r = json.loads(line)
    gold = norm(r["answer"])
    n_q += 1
    d = per_src.setdefault(r["src"], [0, 0, 0, 0])
    d[0] += 1
    kept, seen, ok = [], set(), 0
    for txt in r["samples"]:
        n_samples += 1
        fa = first_answer(txt)
        if fa is None:
            n_no_marker += 1
            continue
        body, ans = fa
        if len(body) < len(txt.strip()):
            n_ran_on += 1
        if ans is None or ans != gold:
            continue
        ok += 1
        n_correct += 1
        key = re.sub(r"\s+", " ", body)
        if key in seen or len(body) > args.max_chars:
            continue
        seen.add(key)
        kept.append(body)
    d[1] += ok
    if ok:
        n_solved += 1
        d[2] += 1
    if args.drop_if_all_correct and ok == len(r["samples"]):
        continue
    for body in kept[: args.keep_per_q]:
        d[3] += 1
        rows.append({
            "question": r["question"],
            "completion": body + fmt.STOP_TOKEN,
            "src": "rft_" + r["src"],
            "n_correct_of_k": ok,
        })

random.Random(args.seed).shuffle(rows)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(args.out.replace(".jsonl", ".check.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps({"text": r["question"] + "\n" + r["completion"]}) + "\n")

stats = {
    "questions": n_q,
    "samples": n_samples,
    "sample_accuracy": n_correct / max(1, n_samples),
    "solved_at_least_once": n_solved,
    "pass_at_k": n_solved / max(1, n_q),
    "samples_without_an_ANSWER_line": n_no_marker,
    "samples_that_ran_on_past_their_answer": n_ran_on,
    "rows": len(rows),
    "per_src": {k: {"q": v[0], "correct_samples": v[1], "solved": v[2], "rows": v[3]}
                for k, v in per_src.items()},
}
print(json.dumps(stats, indent=2))
if args.stats_out:
    json.dump(stats, open(args.stats_out, "w"), indent=2)
print(f"wrote {len(rows)} -> {args.out}")
