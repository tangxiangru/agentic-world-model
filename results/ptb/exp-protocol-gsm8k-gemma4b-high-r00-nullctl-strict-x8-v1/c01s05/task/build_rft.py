#!/usr/bin/env python3
"""Turn sampled generations into rejection-sampling SFT data (correct, deduped, diverse)."""
import argparse
import json
import random
import re


def eq_signature(text):
    """Signature of the arithmetic path, used to keep only distinct reasoning chains."""
    eqs = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return tuple(e.replace(",", "") for e in eqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = []
    n_prob = 0
    n_solved = 0
    for line in open(args.gen):
        r = json.loads(line)
        n_prob += 1
        good = [c for c in r["cands"] if c["correct"] and len(c["text"]) <= args.max_chars]
        if not good:
            continue
        n_solved += 1
        seen = set()
        uniq = []
        rng.shuffle(good)
        # prefer shorter, well-formed solutions
        good.sort(key=lambda c: len(c["text"]))
        for c in good:
            sig = eq_signature(c["text"])
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(c)
            if len(uniq) >= args.max_per_problem:
                break
        for c in uniq:
            body = c["text"].strip()
            body = re.sub(r"\n*ANSWER:\s*[^\n]*$", "", body).strip()
            if not body:
                continue
            out.append({
                "question": r["question"],
                "solution": body,
                "answer": r["answer"],
                "source": "rft",
            })

    rng.shuffle(out)
    with open(args.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"problems={n_prob} solved={n_solved} ({n_solved/max(1,n_prob):.3f}) examples={len(out)}")


if __name__ == "__main__":
    main()
