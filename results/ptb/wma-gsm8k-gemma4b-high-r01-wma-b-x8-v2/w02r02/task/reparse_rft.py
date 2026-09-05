#!/usr/bin/env python3
"""Recover the rejection-sampling corpus from data/rft.raw.jsonl.

Why this exists: in vLLM's offline LLM.generate path the model's
generation_config eos ids are NOT applied (unlike the OpenAI-server path the
grader uses), and skip_special_tokens=True strips <end_of_turn> from the text
before the string stop can match. The model therefore emitted its answer, then
its stop token, then kept going -- 82% of samples carry a run-on tail.

Truncating each sample at the end of its FIRST 'ANSWER: <n>' line reconstructs
exactly the text that would have been returned had the stop id been honoured.
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

STOP_TOKEN = "<end_of_turn>"
FIRST_ANS = re.compile(r"ANSWER:[ \t]*(-?[\d,]+(?:\.\d+)?)")


def norm(v: str) -> str | None:
    v = v.replace(",", "")
    if len(v) > 18:
        return None
    try:
        f = float(v)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(f):
        return None
    return str(int(f)) if f == int(f) else str(f)


def truncate(text: str) -> tuple[str, str] | None:
    """Return (solution up to and including its first ANSWER line, answer)."""
    m = FIRST_ANS.search(text)
    if not m:
        return None
    a = norm(m.group(1))
    if a is None:
        return None
    return text[: m.end()].strip(), a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/rft.raw.jsonl")
    ap.add_argument("--out", default="data/rft.jsonl")
    ap.add_argument("--stats", default="analysis/rft_stats.json")
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kept, n_total, n_correct, solved, n_unparsed = [], 0, 0, 0, 0
    for line in open(args.raw):
        r = json.loads(line)
        gold = r["answer"]
        good = []
        for s in r["samples"]:
            n_total += 1
            t = truncate(s)
            if t is None:
                n_unparsed += 1
                continue
            body, got = t
            if got == gold and len(body) >= args.min_chars:
                n_correct += 1
                good.append(body)
        if good:
            solved += 1
        uniq = sorted(set(good))
        rng.shuffle(uniq)
        for g in uniq[: args.keep_per_problem]:
            kept.append({"system": None, "prompt": r["prompt"],
                         "completion": g + STOP_TOKEN, "answer": gold})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    n_problems = sum(1 for _ in open(args.raw))
    stats = {
        "problems": n_problems,
        "samples": n_total,
        "unparseable": n_unparsed,
        "correct_samples": n_correct,
        "sample_accuracy": n_correct / max(n_total, 1),
        "problems_solved_at_least_once": solved,
        "pass_at_4": solved / max(n_problems, 1),
        "kept_rows": len(kept),
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
