#!/usr/bin/env python3
"""Turn rejection-sampling output into a round-2 SFT file."""
from __future__ import annotations

import argparse
import json
import random
import re

from prep_data import MATH_PROMPT_TEMPLATE, gsm8k_fewshot_raw


ANS_RE = re.compile(r"^ANSWER:\s*(-?[\d,]+(?:\.\d+)?)\s*$", re.M)


def normkey(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())[:400]


def truncate_at_answer(t: str, target: str) -> str | None:
    """Cut the completion right after its first well-formed ANSWER line."""
    m = ANS_RE.search(t)
    if not m:
        return None
    try:
        if format(float(m.group(1).replace(",", "")), ".5g") != format(float(target), ".5g"):
            return None
    except ValueError:
        return None
    return t[:m.end()].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/rft_raw.jsonl")
    ap.add_argument("--out", default="data/rft_sft.jsonl")
    ap.add_argument("--mix", default="data/sft.jsonl")
    ap.add_argument("--mix-n", type=int, default=20000)
    ap.add_argument("--max-easy", type=int, default=2)
    ap.add_argument("--max-hard", type=int, default=4)
    ap.add_argument("--fewshot-frac", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = []
    stats = {"probs": 0, "solved": 0}
    for line in open(args.raw):
        r = json.loads(line)
        stats["probs"] += 1
        clean = [c for c in (truncate_at_answer(t, r["answer"]) for t in r["correct"]) if c]
        if not clean:
            continue
        stats["solved"] += 1
        rate = len(clean) / max(r["n"], 1)
        cap = args.max_easy if rate >= 0.75 else args.max_hard
        seen, kept = set(), []
        cands = sorted(clean, key=len)  # prefer concise, correct chains
        for t in cands:
            k = normkey(t)
            if k in seen:
                continue
            seen.add(k)
            kept.append(t)
            if len(kept) >= cap:
                break
        for t in kept:
            out.append({"question": r["question"], "completion": t})

    print(f"problems={stats['probs']} solved={stats['solved']} "
          f"({stats['solved']/max(stats['probs'],1):.1%}) samples={len(out)}")

    mixed = []
    if args.mix_n > 0:
        allmix = [json.loads(l) for l in open(args.mix)]
        rng.shuffle(allmix)
        mixed = allmix[:args.mix_n]

    fewshot_pool = gsm8k_fewshot_raw()
    rows = []
    for ex in out:
        prompt_body = MATH_PROMPT_TEMPLATE.format(prompt=ex["question"])
        prefix = ""
        if rng.random() < args.fewshot_frac:
            k = rng.randint(1, 5)
            shots = rng.sample(fewshot_pool, k)
            prefix = "\n\n".join(
                f"{q}\n\nReasoning:\n{s}\n\nANSWER: {a}" for q, s, a in shots) + "\n\n"
        rows.append({"prompt": prefix + prompt_body, "completion": ex["completion"],
                     "src": "rft"})
    rows.extend(mixed)
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
