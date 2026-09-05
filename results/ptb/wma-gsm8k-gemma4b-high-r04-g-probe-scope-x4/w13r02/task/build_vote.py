#!/usr/bin/env python3
"""Build in-context self-consistency targets from gen_vllm.py samples.

Each target is three independent attempts at the same problem followed by a
majority vote. The attempts come from real temperature-1.0 samples, so some of
them are wrong on purpose: that is what teaches the vote to do work.

The final 'ANSWER: <n>' line appears exactly once and is the last number in the
target, which is what the grader reads.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from render import END, MATH_PROMPT_TEMPLATE

ANSWER_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def norm(x: str | None) -> str | None:
    if x is None:
        return None
    x = x.replace(",", "").replace("$", "")
    if x.endswith(".0"):
        x = x[:-2]
    return (x.lstrip("0") or "0") if x != "0" else "0"


def strip_answer_line(text: str) -> tuple[str, str] | None:
    """Return (body without its ANSWER line, the answer)."""
    t = text.strip()
    if t.endswith(END):
        t = t[: -len(END)].strip()
    m = ANSWER_RE.search(t)
    if not m:
        return None
    body = t[: m.start()].strip()
    if not body:
        return None
    return body, norm(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-attempts", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    stats = collections.Counter()
    for line in open(args.samples):
        r = json.loads(line)
        stats["questions"] += 1
        gold = norm(str(r["answer"]).strip())
        parsed = []
        for comp in r["completions"]:
            p = strip_answer_line(comp)
            if p and len(p[0]) <= args.max_chars:
                parsed.append(p)
        correct = [p for p in parsed if p[1] == gold]
        wrong = [p for p in parsed if p[1] != gold]
        k = args.n_attempts
        need_correct = k // 2 + 1  # majority must be the gold answer
        if len(correct) < need_correct:
            stats["too_few_correct"] += 1
            continue
        rng.shuffle(correct)
        rng.shuffle(wrong)
        # prefer a genuinely contested vote when the samples offer one
        n_wrong = min(k - need_correct, len(wrong))
        chosen = correct[: k - n_wrong] + wrong[:n_wrong]
        rng.shuffle(chosen)
        stats["contested" if n_wrong else "unanimous"] += 1

        parts = []
        for i, (body, ans) in enumerate(chosen, 1):
            parts.append(f"Attempt {i}:\n{body}\nThis attempt gives {ans}.")
        votes = [a for _, a in chosen]
        tally = ", ".join(votes)
        winner = collections.Counter(votes).most_common(1)[0][0]
        if winner != gold:
            stats["vote_missed_gold"] += 1
            continue
        target = (
            "\n\n".join(parts)
            + f"\n\nThe attempts give {tally}. The most common answer is {winner}."
            + f"\n\nANSWER: {winner}"
        )
        if target.count("ANSWER:") != 1 or norm(ANSWER_RE.search(target).group(1)) != gold:
            stats["format"] += 1
            continue
        rows.append(
            {
                "prompt": MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip()),
                "completion": target + END,
                "answer": gold,
                "source": "synthetic:self-vote",
                "problem": r["question"],
            }
        )
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats["kept"] = len(rows)
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
