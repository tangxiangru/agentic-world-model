#!/usr/bin/env python3
"""Turn gen.py sampling output into rejection-sampled SFT data."""
import argparse, json, re, random, collections

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def path_key(text: str) -> str:
    """Reasoning-path signature: the ordered multiset of equations used."""
    eqs = re.findall(r"[-\d\.,\+\*/\(\)x×÷ ]{3,}=\s*-?[\d,\.]+", text)
    if eqs:
        sig = "|".join(re.sub(r"\s+", "", e) for e in eqs)
    else:
        sig = "|".join(NUM.findall(text))
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-q", type=int, default=4)
    ap.add_argument("--hard-boost", type=int, default=2,
                    help="extra copies allowed for questions with low pass rate")
    ap.add_argument("--min-tokens", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = []
    stats = collections.Counter()
    for line in open(args.input):
        r = json.loads(line)
        cands = [c for c in r["cands"] if c["correct"] and c["ntok"] >= args.min_tokens]
        n_tot = len(r["cands"])
        pass_rate = len(cands) / max(1, n_tot)
        if not cands:
            stats["no_correct"] += 1
            continue
        # must end with the required ANSWER line
        cands = [c for c in cands
                 if re.search(r"ANSWER:\s*\$?-?[\d,]+(\.\d+)?\s*\.?$", c["text"].strip())]
        if not cands:
            stats["bad_format"] += 1
            continue
        cap = args.max_per_q
        if pass_rate <= 0.5:
            cap += args.hard_boost
        # diverse selection by reasoning path
        by_path = {}
        rng.shuffle(cands)
        for c in cands:
            k = path_key(c["text"])
            by_path.setdefault(k, []).append(c)
        chosen = []
        for k, v in by_path.items():
            v.sort(key=lambda c: c["ntok"])
            chosen.append(v[len(v) // 2])
        chosen.sort(key=lambda c: c["ntok"])
        chosen = chosen[:cap]
        stats["kept_q"] += 1
        stats["kept_sol"] += len(chosen)
        for c in chosen:
            out.append({"question": r["question"], "response": c["text"].strip(),
                        "answer": r["target"], "source": "rft"})
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(dict(stats), "->", len(out), args.out)


if __name__ == "__main__":
    main()
