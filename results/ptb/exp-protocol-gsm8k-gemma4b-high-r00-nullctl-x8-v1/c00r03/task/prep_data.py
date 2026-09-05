#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 (+ optional extras)."""
import argparse, glob, json, os, random, re

OMI_DIR = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data"

BOX_RE = re.compile(r"\\boxed\{")


def strip_boxed(sol: str) -> str:
    """Replace \\boxed{X} with X (last occurrence handling nested braces)."""
    out = []
    i = 0
    while True:
        m = BOX_RE.search(sol, i)
        if not m:
            out.append(sol[i:])
            break
        out.append(sol[i:m.start()])
        j = m.end()
        depth = 1
        start = j
        while j < len(sol) and depth:
            if sol[j] == "{":
                depth += 1
            elif sol[j] == "}":
                depth -= 1
            j += 1
        out.append(sol[start:j - 1])
        i = j
    return "".join(out)


def clean_solution(sol: str, answer: str) -> str | None:
    sol = strip_boxed(sol).strip()
    if not sol:
        return None
    # drop trailing 'The answer is ...' style lines, we add our own
    lines = [l.rstrip() for l in sol.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    sol = "\n".join(lines).strip()
    return sol + "\n\nANSWER: " + answer.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--n-gsm", type=int, default=120000)
    ap.add_argument("--n-math", type=int, default=30000)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    files = sorted(glob.glob(os.path.join(OMI_DIR, "train_1M-*.parquet")))
    gsm, mth = [], []
    for f in files:
        df = pd.read_parquet(f)
        for src, bucket in (("gsm8k", gsm), ("math", mth)):
            sub = df[df["problem_source"].isin([src, "augmented_" + src])]
            bucket.extend(sub.to_dict("records"))
        del df
    print("pool sizes", len(gsm), len(mth))

    rng = random.Random(args.seed)
    rng.shuffle(gsm)
    rng.shuffle(mth)

    recs = []
    seen = set()
    for bucket, n, tag in ((gsm, args.n_gsm, "gsm"), (mth, args.n_math, "math")):
        kept = 0
        for r in bucket:
            if kept >= n:
                break
            q = str(r["problem"]).strip()
            a = str(r["expected_answer"]).strip()
            s = str(r["generated_solution"])
            if not q or not a or len(a) > 40:
                continue
            if len(q) + len(s) > args.max_chars:
                continue
            key = (q, a)
            if key in seen:
                continue
            sol = clean_solution(s, a)
            if sol is None or "\\boxed" in sol:
                continue
            seen.add(key)
            recs.append({"question": q, "solution": sol, "answer": a, "src": tag})
            kept += 1
        print(tag, "kept", kept)

    rng.shuffle(recs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(recs), "->", args.out)


if __name__ == "__main__":
    main()
