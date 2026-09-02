#!/usr/bin/env python3
"""Post-filter for gen_rft.py output.

gen_rft.py kept a sample if the LAST numeric word matched gold, which is exactly the
grader's rule - but it is not a sufficient rule for TRAINING data. At temperature 1.0 the
model still sometimes runs past its own answer and invents follow-up problems; such a
completion can end on the right number by luck and carries dozens of 'ANSWER:' lines.
Counted over data/rft_v1.jsonl: only 7859 of 18328 rows had exactly one marker, and one
row had 102 (pitfall double_answer_format, and a direct teacher of the exp-01 failure).

This script keeps the chain up to and including the FIRST 'ANSWER:' line, requires the
number on that line to be gold, strips any '#### n' second marker out of the body, and
emits one target with exactly one answer marker ending in the stop token.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

STOP = "<end_of_turn>"
ANS = re.compile(r"(?m)^[ \t]*ANSWER:[ \t]*(.+?)[ \t]*$")
HASH = re.compile(r"(?m)^[ \t]*####.*$")


def norm(x: str) -> str:
    x = x.strip().rstrip(".").replace(",", "").replace("$", "")
    if x.endswith(".0"):
        x = x[:-2]
    return x


ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-per-question", type=int, default=4)
ap.add_argument("--stats", default=None)
a = ap.parse_args()

reasons = Counter()
kept, seen, per_q = [], set(), Counter()
for line in open(a.inp):
    r = json.loads(line)
    txt = r["completion"][: -len(STOP)] if r["completion"].endswith(STOP) else r["completion"]
    m = ANS.search(txt)
    if not m:
        reasons["no_answer_line"] += 1
        continue
    if norm(m.group(1)) != norm(r["answer"]):
        reasons["first_answer_wrong"] += 1
        continue
    body = HASH.sub("", txt[: m.start()]).rstrip()
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) < 30:
        reasons["body_too_short"] += 1
        continue
    if "ANSWER:" in body:
        reasons["marker_in_body"] += 1
        continue
    completion = f"{body}\n\nANSWER: {norm(r['answer'])}{STOP}"
    key = re.sub(r"\s+", " ", r["question"] + "||" + body)
    if key in seen:
        reasons["duplicate"] += 1
        continue
    if per_q[r["question"]] >= a.max_per_question:
        reasons["over_cap"] += 1
        continue
    seen.add(key)
    per_q[r["question"]] += 1
    kept.append({**r, "completion": completion, "target": completion})

with open(a.out, "w") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")

stats = {"in": sum(reasons.values()) + len(kept), "kept": len(kept),
         "dropped": dict(reasons), "distinct_questions": len(per_q)}
print(json.dumps(stats, indent=2))
if a.stats:
    json.dump(stats, open(a.stats, "w"), indent=2)
