#!/usr/bin/env python3
"""SFT corpus v2: maximise unique problems rather than solutions per problem.

v1 sampled up to 2 solutions from a 1M slice of OpenMathInstruct-2 and trained
2 epochs. The same compute buys ~2x the unique problems at 1 solution each,
which is usually the better trade for a 4B model. Sources are unchanged
(gsm8k TRAIN + OMI2 rows whose problem_source contains "gsm8k"); the probe
questions are still excluded and the gsm8k test split is still untouched.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from build_sft_data import (END_OF_TURN, RAW, ROOT, build_user, clean_number,
                            gsm8k_rows, strip_boxed_tail)


def omi2_best(path: Path, per_problem: int, max_chars: int, rng: random.Random):
    by_problem: dict[str, list[dict]] = {}
    for line in path.open():
        r = json.loads(line)
        gold = clean_number(str(r["expected_answer"]))
        if gold is None:
            continue
        sol = strip_boxed_tail(r["generated_solution"])
        if not sol or "\\boxed" in sol or "####" in sol:
            continue
        if len(sol) < 60 or len(sol) > max_chars:
            continue
        by_problem.setdefault(r["problem"].strip(), []).append(
            {"question": r["problem"].strip(),
             "solution": f"{sol}\n\nANSWER: {gold}",
             "gold": gold, "src": r["problem_source"]})
    out = []
    for cands in by_problem.values():
        seen, uniq = set(), []
        for c in cands:
            if c["solution"] in seen:
                continue
            seen.add(c["solution"])
            uniq.append(c)
        rng.shuffle(uniq)
        out.extend(uniq[:per_problem])
    return out, len(by_problem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi2", default=str(RAW / "omi2_gsm8k_2m.jsonl"))
    ap.add_argument("--per-problem", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "data" / "sft_v2.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=str(ROOT / "data" / "probe250.jsonl"))
    args = ap.parse_args()

    held = {json.loads(l)["question"].strip() for l in Path(args.exclude).open()}
    rng = random.Random(args.seed)
    g = [r for r in gsm8k_rows(RAW / "gsm8k_train.jsonl") if r["question"] not in held]
    o, n_prob = omi2_best(Path(args.omi2), args.per_problem, args.max_chars, rng)
    o = [r for r in o if r["question"] not in held]
    print(f"gsm8k rows {len(g)}; omi2 unique problems {n_prob}; omi2 rows kept {len(o)}")

    rows = g * args.gsm8k_repeat + o
    rng.shuffle(rows)
    with Path(args.out).open("w") as f:
        for r in rows:
            f.write(json.dumps({"question": r["question"],
                                "prompt": build_user(r["question"]),
                                "completion": r["solution"] + END_OF_TURN,
                                "gold": r["gold"], "src": r["src"]}) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(Counter(r["src"] for r in rows))


if __name__ == "__main__":
    main()
