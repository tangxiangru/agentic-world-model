#!/usr/bin/env python3
"""Build two SFT sets from dumped policy samples:

  --mode rft : one correct solution per problem (expert iteration)
  --mode sc  : three sampled attempts + an explicit majority vote, so the model
               performs self-consistency inside a single response
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from prep_data import MATH_PROMPT_TEMPLATE, gsm8k_fewshot_blocks

ANS_LINE = re.compile(r"\n*ANSWER:[^\n]*$")


def body_of(text: str) -> str:
    return ANS_LINE.sub("", text).strip()


def fmt_num(x: float) -> str:
    if x == int(x):
        return str(int(x))
    return ("%f" % x).rstrip("0").rstrip(".")


def build_sc(rec, rng, n_attempts=3):
    cands = [c for c in rec["cands"] if c["ans"] is not None and body_of(c["text"])]
    if len(cands) < n_attempts:
        return None
    counts = Counter(c["ans"] for c in cands)
    gold_ans = next((c["ans"] for c in cands if c["ok"]), None)
    if gold_ans is None:
        return None
    top = counts.most_common()
    if top[0][0] != gold_ans or (len(top) > 1 and top[1][1] == top[0][1]):
        return None  # gold must be the strict mode

    right = [c for c in cands if c["ok"]]
    wrong = [c for c in cands if not c["ok"]]
    if wrong and rng.random() < 0.6:
        chosen = rng.sample(right, n_attempts - 1) + [rng.choice(wrong)]
        rng.shuffle(chosen)
    else:
        if len(right) < n_attempts:
            return None
        chosen = rng.sample(right, n_attempts)

    parts = []
    results = []
    for i, c in enumerate(chosen, 1):
        results.append(fmt_num(c["ans"]))
        parts.append(f"Attempt {i}:\n{body_of(c['text'])}\nResult: {results[-1]}")
    gold_str = rec["answer"].strip()
    listed = ", ".join(results)
    tally = Counter(results)
    if tally[fmt_num(gold_ans)] == len(results):
        verdict = f"All {len(results)} attempts agree on {gold_str}."
    else:
        verdict = (
            f"The attempts give {listed}. "
            f"{tally[fmt_num(gold_ans)]} of {len(results)} agree on {gold_str}, "
            "so that is the answer."
        )
    return "\n\n".join(parts) + f"\n\n{verdict}\n\nANSWER: {gold_str}"


def build_rft(rec, rng, max_keep=2):
    good, seen = [], set()
    for c in rec["cands"]:
        if not c["ok"]:
            continue
        key = re.sub(r"\s+", " ", c["text"])
        if key in seen:
            continue
        seen.add(key)
        good.append(c["text"])
    good.sort(key=len)
    return good[:max_keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["sc", "rft"], required=True)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fewshot_pool = gsm8k_fewshot_blocks(600, 0)
    rows = []
    with open(args.inp) as f:
        for line in f:
            rec = json.loads(line)
            if args.mode == "sc":
                doc = build_sc(rec, rng)
                if doc:
                    rows.append((rec["problem"], doc, rec["answer"], rec["source"]))
            else:
                for doc in build_rft(rec, rng):
                    rows.append((rec["problem"], doc, rec["answer"], rec["source"]))
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for problem, doc, ans, src in rows:
            system = ""
            if rng.random() < args.fewshot_frac:
                k = rng.randint(2, 10)
                system = "\n\n".join(rng.sample(fewshot_pool, k))
            f.write(
                json.dumps(
                    {
                        "system": system,
                        "user": MATH_PROMPT_TEMPLATE.format(prompt=problem),
                        "assistant": doc,
                        "answer": ans,
                        "source": src,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
