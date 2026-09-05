#!/usr/bin/env python3
"""Turn rejection-sampling output into an SFT file in the grader's format."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fmt import EOT, render_prompt, render_target  # noqa: E402


def norm(s: str) -> str:
    s = re.sub(r"\d+", "#", s.lower())
    return re.sub(r"\W+", "", s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-q", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=3500)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from build_fewshot import fewshot_system_message
    system = fewshot_system_message()

    rows = []
    n_q = n_used = 0
    for line in open(args.samples):
        r = json.loads(line)
        n_q += 1
        sols = [s for s in r["solutions"]
                if 20 < len(s) <= args.max_chars
                and EOT not in s and "<start_of_turn>" not in s
                and s.count("ANSWER:") == 1]
        if not sols:
            continue
        # prefer distinct reasoning paths, shortest first (fewer wandering chains)
        sols.sort(key=len)
        picked, seen = [], set()
        for s in sols:
            k = norm(s)
            if k in seen:
                continue
            seen.add(k)
            picked.append(s)
            if len(picked) >= args.max_per_q:
                break
        n_used += 1
        for s in picked:
            rows.append((r["q"], s, r["src"]))

    rng.shuffle(rows)
    n_fs = 0
    with open(args.out, "w") as f:
        for q, sol, src in rows:
            use_fs = rng.random() < args.fewshot_frac
            n_fs += int(use_fs)
            f.write(json.dumps({
                "prompt": render_prompt(q, system if use_fs else None),
                "completion": render_target(sol),
                "target": render_target(sol),
                "source": "rft:" + src,
                "fewshot": use_fs,
            }) + "\n")
    print(f"questions {n_q}, with a kept solution {n_used}, rows {len(rows)} "
          f"({n_fs} few-shot) -> {args.out}")


if __name__ == "__main__":
    main()
