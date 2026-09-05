"""Combine the exp-02 SFT corpus with the exp-04 rejection-sampled corpus."""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import EOT, is_correct  # noqa: E402

# The rejection-sampling run dropped the string stop condition, so vLLM did not
# halt on <end_of_turn> and 4490 of 20840 kept chains carry a degenerate tail
# after their answer (the answer repeated, or a stray 'model' turn). The chain
# up to and including the first ANSWER line is still a verified-correct
# solution, so truncate there instead of throwing the row away.
FIRST_ANSWER = re.compile(r"^(.*?ANSWER:[ \t]*-?[\d,]+(?:\.\d+)?)", re.DOTALL)


def clean_rft(row: dict) -> dict | None:
    body = row["completion"]
    if body.endswith(EOT):
        body = body[: -len(EOT)]
    m = FIRST_ANSWER.match(body)
    if not m:
        return None
    text = m.group(1).strip()
    if text.count("ANSWER:") != 1 or not is_correct(text, row["answer"]):
        return None
    return {**row, "completion": text + EOT}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="/home/ben/task/data/sft_train.jsonl")
    ap.add_argument("--rft", default="/home/ben/task/data/rft_raw.jsonl")
    ap.add_argument("--out", default="/home/ben/task/data/mix_train.jsonl")
    ap.add_argument("--docs-out", default="/home/ben/task/data/mix_train_docs.jsonl")
    ap.add_argument("--max-rft", type=int, default=40000)
    ap.add_argument("--max-sft", type=int, default=54197)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sft = [json.loads(l) for l in open(args.sft)]
    raw = [json.loads(l) for l in open(args.rft)]
    rft = [c for c in (clean_rft(r) for r in raw) if c is not None]
    n_trunc = sum(
        1
        for r, c in zip(raw, (clean_rft(r) for r in raw))
        if c is not None and len(c["completion"]) < len(r["completion"])
    )
    print(f"[rft] {len(raw)} raw -> {len(rft)} usable ({n_trunc} truncated at their first ANSWER line)")
    rng.shuffle(sft)
    rng.shuffle(rft)
    rows = sft[: args.max_sft] + rft[: args.max_rft]
    rng.shuffle(rows)

    src = collections.Counter(r.get("src", "sft") for r in rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {"prompt": r["prompt"], "completion": r["completion"], "answer": r["answer"]}
                )
                + "\n"
            )
    with open(args.docs_out, "w") as f:
        for r in rows:
            # prompt carries the problem statement; completion carries the chain
            f.write(json.dumps({"text": r["prompt"] + "\n" + r["completion"]}) + "\n")
    print(json.dumps({"total": len(rows), "by_source": dict(src), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
