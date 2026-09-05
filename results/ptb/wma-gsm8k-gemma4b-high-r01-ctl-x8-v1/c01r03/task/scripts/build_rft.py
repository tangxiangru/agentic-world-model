#!/usr/bin/env python3
"""Turn sample_model.py --mode rft output into SFT rows (rejection sampling).

Keeps only samples the grader itself would mark correct, deduplicates, and caps
the number kept per question so easy questions do not swamp hard ones.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmt import ANSWER_MARKER, END_OF_TURN, render_prompt_fast  # noqa: E402
from eval_format import build_system_message, build_user_message  # noqa: E402

EQ = re.compile(r"[-+*/=]")


def signature(text: str) -> str:
    """Coarse reasoning fingerprint: the sequence of numbers used.

    The RFT paper keeps solutions with distinct equation sets; this is a cheap
    stand-in that still throws away restatements of the same arithmetic path.
    """
    nums = re.findall(r"\d+\.?\d*", text)
    return "|".join(nums)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-q", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--max-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    n_q = n_q_with_any = 0
    for line in open(args.samples):
        r = json.loads(line)
        n_q += 1
        keep = []
        seen_sig = set()
        cands = [s for s in r["samples"] if s["correct"] and s["finish"] == "stop"]
        # shortest first: shorter correct chains are usually the cleaner ones
        cands.sort(key=lambda s: len(s["text"]))
        for s in cands:
            t = s["text"].strip()
            if t.count(ANSWER_MARKER) != 1:
                continue
            if len(t) > args.max_chars:
                continue
            sig = signature(t)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            keep.append(t)
            if len(keep) >= args.max_per_q:
                break
        if keep:
            n_q_with_any += 1
        for t in keep:
            rows.append({"question": r["question"], "target": t})

    rng.shuffle(rows)
    system = build_system_message()
    n_few = int(len(rows) * args.fewshot_frac)
    out = []
    for i, r in enumerate(rows):
        sysm = system if i < n_few else None
        out.append(
            {
                "prompt": render_prompt_fast(sysm, build_user_message(r["question"])),
                "completion": r["target"] + END_OF_TURN,
                "source": "rft_self",
                "fewshot": sysm is not None,
            }
        )
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")
    print(
        json.dumps(
            {
                "questions": n_q,
                "questions_with_a_correct_sample": n_q_with_any,
                "coverage": round(n_q_with_any / max(1, n_q), 4),
                "rows": len(out),
                "fewshot_rows": n_few,
                "out": args.out,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
