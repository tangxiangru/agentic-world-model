#!/usr/bin/env python3
"""Build GSM8K-style SFT data from nvidia/OpenMathInstruct-2 (gsm8k-derived rows).

Target format matches the grader: chain of thought, then a final line
"ANSWER: <number>". The stop token is appended by the trainer.
"""
import argparse, glob, json, random, re, collections
import pyarrow.parquet as pq

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    if not NUMERIC.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    if a.startswith("+"):
        a = a[1:]
    return a


def strip_boxed(sol: str) -> str:
    # \boxed{45} -> 45 ; leave everything else intact
    return BOXED.sub(lambda m: m.group(1), sol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem-aug", type=int, default=1)
    ap.add_argument("--max-per-problem-orig", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--max-sol-chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    per_problem = collections.Counter()
    stats = collections.Counter()
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        d = t.to_pydict()
        for q, sol, ans, src in zip(d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]):
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            stats["gsm8k_rows"] += 1
            a = clean_answer(ans)
            if a is None:
                stats["drop_nonnumeric"] += 1
                continue
            if len(sol) > args.max_sol_chars or len(q) > 1500:
                stats["drop_long"] += 1
                continue
            cap = args.max_per_problem_orig if src == "gsm8k" else args.max_per_problem_aug
            if per_problem[q] >= cap:
                stats["drop_cap"] += 1
                continue
            body = strip_boxed(sol).strip()
            # the final number the grader reads must be the ANSWER line, and the
            # marker must appear exactly once
            if "ANSWER:" in body or "####" in body:
                stats["drop_marker"] += 1
                continue
            if "\\[" in body or "\\begin" in body:
                stats["drop_latex"] += 1
                continue
            per_problem[q] += 1
            target = body + "\nANSWER: " + a + "<end_of_turn>"
            rows.append({"question": q.strip(), "solution": body, "answer": a,
                         "target": target, "source": src})
            stats["kept"] += 1
    rng.shuffle(rows)
    rows = rows[: args.max_rows]
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    stats["written"] = len(rows)
    stats["src_gsm8k"] = sum(1 for r in rows if r["source"] == "gsm8k")
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
