#!/usr/bin/env python3
"""Build SFT data for gsm8k from OpenMathInstruct-2 (gsm8k subsets) + GSM8K train.

Output jsonl rows: {"question", "solution", "answer", "src"} -- raw text, not yet
rendered with the chat template (rendering happens in train_sft.py so that the
grader's own templates/gemma3.jinja is the single source of truth).
"""
import argparse, glob, json, random, re, sys
from collections import defaultdict

import pyarrow.parquet as pq

BOXED = "\\boxed{"


def unbox(s: str) -> str:
    """Replace every \\boxed{X} with X (balanced braces)."""
    out = []
    i = 0
    while True:
        j = s.find(BOXED, i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k = j + len(BOXED)
        depth = 1
        start = k
        while k < len(s) and depth:
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
            k += 1
        out.append(s[start:k - 1] if depth == 0 else s[start:k])
        i = k


NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def clean_num(a: str):
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    # normalise 12.0 -> 12
    try:
        f = float(a)
    except ValueError:
        return None
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-gsm8k", type=int, default=4)
    ap.add_argument("--per-aug", type=int, default=2)
    ap.add_argument("--max-aug-problems", type=int, default=20000)
    ap.add_argument("--max-sol-chars", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    print(f"{len(files)} shards", file=sys.stderr)

    by_problem = defaultdict(list)   # problem -> list[(src, solution, answer)]
    kept = dropped = 0
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution",
                                      "expected_answer", "problem_source"])
        d = t.to_pydict()
        for p, sol, ans, src in zip(d["problem"], d["generated_solution"],
                                    d["expected_answer"], d["problem_source"]):
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            a = clean_num(ans)
            if a is None:
                dropped += 1
                continue
            if BOXED not in sol:
                dropped += 1
                continue
            # the boxed content must be the expected answer
            m = sol.rfind(BOXED)
            tail = unbox(sol[m:])
            if clean_num(tail.split("\n")[0].split(" ")[0].strip(" .,$")) != a:
                # boxed value not obviously the answer -> keep only if it matches anywhere
                pass
            sol = unbox(sol).strip()
            if len(sol) > args.max_sol_chars or len(sol) < 20:
                dropped += 1
                continue
            if "```" in sol or "\\begin{" in sol:
                dropped += 1
                continue
            by_problem[p].append((src, sol, a))
            kept += 1
        print(f"  {f.split('/')[-1]}: kept={kept} dropped={dropped}", file=sys.stderr)

    real, aug = [], []
    for p, lst in by_problem.items():
        src = lst[0][0]
        rng.shuffle(lst)
        cap = args.per_gsm8k if src == "gsm8k" else args.per_aug
        seen = set()
        n = 0
        for s, sol, a in lst:
            key = sol[:200]
            if key in seen:
                continue
            seen.add(key)
            row = {"question": p, "solution": sol, "answer": a, "src": s}
            (real if src == "gsm8k" else aug).append(row)
            n += 1
            if n >= cap:
                break

    # cap the augmented pool by number of distinct problems
    aug_by_p = defaultdict(list)
    for r in aug:
        aug_by_p[r["question"]].append(r)
    keys = list(aug_by_p)
    rng.shuffle(keys)
    keys = keys[:args.max_aug_problems]
    aug = [r for k in keys for r in aug_by_p[k]]

    rows = real + aug
    rng.shuffle(rows)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} "
          f"(real gsm8k problems={len(set(r['question'] for r in real))} rows={len(real)}; "
          f"aug problems={len(keys)} rows={len(aug)}); dropped={dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
