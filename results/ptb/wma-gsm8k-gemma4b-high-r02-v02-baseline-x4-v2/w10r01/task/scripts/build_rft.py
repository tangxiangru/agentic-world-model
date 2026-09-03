#!/usr/bin/env python3
"""Turn scripts/gen.py samples into rejection-sampling fine-tuning rows.

Keeps only completions whose graded answer matches gold (the same last-number
rule the harness applies), caps how many survive per question, and prefers
distinct reasoning paths over near-duplicates of the same chain.
"""
from __future__ import annotations

import argparse
import json
import re

from common import ANSWER_MARKER, extract_answer, graded_correct, render_prompt, render_target


def path_key(text: str) -> str:
    """Signature of a reasoning chain: the sequence of numbers it computes."""
    body = text.split(ANSWER_MARKER)[0]
    return ",".join(re.findall(r"-?\d[\d,]*\.?\d*", body)[:24])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument(
        "--hard-threshold",
        type=int,
        default=0,
        help="questions solved at most this many times out of n keep --max-per-question "
        "paths; everything easier keeps 1. 0 disables the distinction.",
    )
    ap.add_argument("--max-chars", type=int, default=3000)
    args = ap.parse_args()

    rows = []
    n_q = n_solved = n_cand = 0
    for path in args.samples:
        for line in open(path):
            r = json.loads(line)
            n_q += 1
            gold = str(r["gold"])
            keep, seen = [], set()
            for comp, fin in zip(r["completions"], r["finish"]):
                if fin != "stop":
                    continue  # never train on a truncated chain
                comp = comp.strip()
                if len(comp) > args.max_chars:
                    continue
                if comp.count(ANSWER_MARKER) != 1 or "####" in comp:
                    continue
                if not graded_correct(comp, gold):
                    continue
                k = path_key(comp)
                if k in seen:
                    continue
                seen.add(k)
                keep.append(comp)
            n_cand += len(keep)
            if not keep:
                continue
            n_solved += 1
            # shortest first: the terse correct chain is the one to reinforce.
            # Spend the extra path only on questions the model rarely solves;
            # a question it gets right 4/4 needs no reinforcement at all.
            keep.sort(key=len)
            n_correct = sum(1 for c, f in zip(r["completions"], r["finish"])
                            if f == "stop" and graded_correct(c, gold))
            cap = args.max_per_question
            if args.hard_threshold and n_correct > args.hard_threshold:
                cap = 1
            for comp in keep[:cap]:
                body = comp.split(ANSWER_MARKER)[0].strip()
                rows.append(
                    {
                        "prompt": render_prompt(r["question"]),
                        "target": render_target(body, extract_answer(comp)),
                        "src": "rft:self",
                        "question": r["question"],
                        "answer": str(extract_answer(comp)),
                    }
                )

    bad = [r for r in rows if r["target"].count(ANSWER_MARKER) != 1 or not r["target"].endswith("<end_of_turn>")]
    assert not bad, bad[:1]
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(
        json.dumps(
            {
                "questions": n_q,
                "solved_at_least_once": n_solved,
                "solve_rate": round(n_solved / max(1, n_q), 4),
                "distinct_correct_paths": n_cand,
                "rows_written": len(rows),
                "out": args.out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
