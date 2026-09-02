#!/usr/bin/env python3
"""Stage-3 training mix: round-2 RFT (from sft2) + a slice of round-1 RFT + original GSM8K."""
from __future__ import annotations

import json
import random
from collections import defaultdict


def load(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def split_cap(path, gsm_qs, cap_gsm, cap_aug, max_aug, rng):
    by_q = defaultdict(list)
    for r in load(path):
        by_q[r["question"]].append(r)
    g, a = [], []
    for q, rs in by_q.items():
        cap = cap_gsm if q in gsm_qs else cap_aug
        keep = rs if len(rs) <= cap else rng.sample(rs, cap)
        (g if q in gsm_qs else a).extend(keep)
    rng.shuffle(a)
    return g, a[:max_aug]


def clean(sol: str) -> str:
    return "\n".join(l for l in sol.split("\n") if not l.strip().startswith("####")).strip()


def main():
    rng = random.Random(0)
    gsm = load("data/gsm8k_train.jsonl")
    gsm_qs = {r["question"] for r in gsm}

    g2, a2 = split_cap("data/rft2.jsonl", gsm_qs, 3, 2, 24000, rng)
    g1, a1 = split_cap("data/rft1.jsonl", gsm_qs, 1, 1, 8000, rng)
    mm = load("data/metamath_gsm.jsonl", 4000)

    mix = g2 + a2 + g1 + a1 + gsm + mm
    out = []
    for r in mix:
        s = clean(r["solution"])
        if s:
            r = dict(r, solution=s)
            out.append(r)
    rng.shuffle(out)
    with open("data/mix3.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    with open("data/mix3_decon.jsonl", "w") as f:
        for i, r in enumerate(out):
            f.write(json.dumps({"id": i, "text": r["question"] + "\n" + r["solution"]
                                + "\nANSWER: " + r["answer"]}) + "\n")
    print(f"rft2_gsm={len(g2)} rft2_aug={len(a2)} rft1_gsm={len(g1)} rft1_aug={len(a1)} "
          f"gsm={len(gsm)} mm={len(mm)} total={len(out)}")


if __name__ == "__main__":
    main()
