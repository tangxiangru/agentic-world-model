#!/usr/bin/env python3
"""Build the GSM8K-style SFT set from OpenMathInstruct-2 (gsm8k-derived rows only).

Output jsonl rows: {"problem": str, "completion": str, "final_answer": str, "source": str}
`completion` is the exact training target: solution body, then the single answer
marker line "ANSWER: <int>", then the "<end_of_turn>" terminator that vLLM stops on.
"""
import argparse
import glob
import hashlib
import json
import random
import re
import collections

SNAP = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/469216e3f46f4dacf476b382e192485ea51a143e"
INT_RE = re.compile(r"^-?\d+$")
ANSWER_MARK_RE = re.compile(r"ANSWER\s*:", re.IGNORECASE)


def unbox(sol: str):
    """Replace the single \\boxed{...} with its contents. Returns None if not exactly one."""
    i = sol.find("\\boxed{")
    if i < 0 or sol.find("\\boxed{", i + 1) >= 0:
        return None
    j = i + len("\\boxed{")
    depth = 1
    k = j
    while k < len(sol):
        if sol[k] == "{":
            depth += 1
        elif sol[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    if depth != 0:
        return None
    return sol[:i] + sol[j:k] + sol[k + 1:]


def clean_tail(body: str) -> str:
    """Trim trailing latex scaffolding left behind by unboxing."""
    body = body.rstrip()
    # drop a dangling closing display-math delimiter with nothing numeric before it
    for junk in ("\\]", "\\)", "$$", "$"):
        if body.endswith(junk):
            stripped = body[: -len(junk)].rstrip()
            if stripped:
                body = stripped
    return body.rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-gsm8k", type=int, default=8, help="max solutions per original gsm8k problem")
    ap.add_argument("--cap-aug", type=int, default=2, help="max solutions per augmented problem")
    ap.add_argument("--max-rows", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pattern", default="train_1M-*.parquet")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{SNAP}/data/{args.pattern}"))
    assert files, "OpenMathInstruct-2 parquet files not found"

    rows = []
    stats = collections.Counter()
    for f in files:
        t = pq.read_table(f)
        d = t.to_pydict()
        for problem, sol, ans, src in zip(
            d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]
        ):
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            stats["gsm8k_rows"] += 1
            if not INT_RE.match(ans.strip()):
                stats["drop_non_int_answer"] += 1
                continue
            body = unbox(sol)
            if body is None:
                stats["drop_boxed"] += 1
                continue
            body = clean_tail(body)
            if ANSWER_MARK_RE.search(body) or "####" in body:
                stats["drop_marker_in_body"] += 1
                continue
            if len(body) < 40:
                stats["drop_short"] += 1
                continue
            rows.append(
                {
                    "problem": problem.strip(),
                    "completion": body + "\nANSWER: " + ans.strip() + "<end_of_turn>",
                    "final_answer": ans.strip(),
                    "source": src,
                }
            )

    # cap solutions per problem, dedup identical (problem, solution)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    per_problem = collections.Counter()
    seen = set()
    kept = []
    for r in rows:
        key = hashlib.md5(r["problem"].encode()).hexdigest()
        cap = args.cap_gsm8k if r["source"] == "gsm8k" else args.cap_aug
        if per_problem[key] >= cap:
            stats["drop_cap"] += 1
            continue
        sig = hashlib.md5((key + r["completion"]).encode()).hexdigest()
        if sig in seen:
            stats["drop_dup"] += 1
            continue
        seen.add(sig)
        per_problem[key] += 1
        kept.append(r)

    rng.shuffle(kept)
    kept = kept[: args.max_rows]
    with open(args.out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    stats["kept"] = len(kept)
    stats["unique_problems"] = len(per_problem)
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
