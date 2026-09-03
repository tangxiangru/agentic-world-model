"""Build the SFT corpus for GSM8K from OpenMathInstruct-2 + the GSM8K train split.

Everything is rendered into the grader's exact prompt/completion strings by
scripts/fmt.py. Output: jsonl rows {prompt, completion, text, src}.

No GSM8K *test* item is read anywhere in this file. OpenMathInstruct-2's
`gsm8k` / `augmented_gsm8k` rows are seeded from the GSM8K *train* split
(Toshniwal et al. 2024); the result is still run through contamination_check.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import hashlib

import pyarrow.parquet as pq
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

OMI_DIR = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data"

INT_RE = re.compile(r"^-?\d+$")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def norm_q(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()


def acceptable_answer(a: str, numeric_only: bool) -> bool:
    a = a.strip()
    if not a:
        return False
    if numeric_only:
        return bool(NUM_RE.match(a))
    return len(a) <= 40


def load_omi(shards, want, cap_per_problem, numeric_only=True, exclude=None):
    """Yield rows grouped by problem_source with a cap on solutions per problem."""
    kept = defaultdict(list)
    seen = defaultdict(int)
    exclude = exclude or set()
    for sh in shards:
        path = os.path.join(OMI_DIR, f"train-{sh:05d}-of-00032.parquet")
        if not os.path.exists(path):
            print("missing shard", path)
            continue
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=20000, columns=["problem", "generated_solution", "expected_answer", "problem_source"]):
            for r in batch.to_pylist():
                src = r["problem_source"]
                if src not in want:
                    continue
                if len(kept[src]) >= want[src]:
                    continue
                ans = (r["expected_answer"] or "").strip()
                if not acceptable_answer(ans, numeric_only):
                    continue
                key = (src, norm_q(r["problem"]))
                if seen[key] >= cap_per_problem:
                    continue
                if exclude and hashlib.md5(
                        (r["problem"] + "\x00" + r["generated_solution"]).encode()).hexdigest() in exclude:
                    continue
                tgt = fmt.make_target(r["generated_solution"], ans)
                if tgt is None:
                    continue
                seen[key] += 1
                kept[src].append({"question": r["problem"].strip(), "target": tgt, "src": src})
        print("after shard", sh, {k: len(v) for k, v in kept.items()}, flush=True)
        if all(len(kept[k]) >= want[k] for k in want):
            break
    return kept


def load_gsm8k_train():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        sol, ans = r["answer"].rsplit("####", 1)
        ans = ans.strip()
        if not INT_RE.match(ans.replace(",", "")):
            continue
        tgt = fmt.make_target(sol, ans.replace(",", ""))
        if tgt is None:
            continue
        out.append({"question": r["question"].strip(), "target": tgt, "src": "gsm8k_train_gold"})
    return out


def fewshot_pool():
    """Few-shot exemplars for the small fraction of rows trained with a prefix.

    Drawn from the GSM8K TRAIN split, same rendering the harness uses.
    """
    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for r in ds:
        reasoning, ans = r["answer"].rsplit("####", 1)
        pool.append(f"{r['question'].strip()}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: {ans.strip()}")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm8k", type=int, default=40000)
    ap.add_argument("--n-aug-gsm8k", type=int, default=110000)
    ap.add_argument("--n-math", type=int, default=8000)
    ap.add_argument("--n-aug-math", type=int, default=22000)
    ap.add_argument("--cap-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.06)
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-questions", default=None,
                    help="jsonl of rows already used; their questions are skipped entirely")
    ap.add_argument("--no-gsm8k-gold", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    want = {
        "gsm8k": args.n_gsm8k,
        "augmented_gsm8k": args.n_aug_gsm8k,
        "math": args.n_math,
        "augmented_math": args.n_aug_math,
    }
    want = {k: v for k, v in want.items() if v > 0}
    seen_q = set()
    if args.exclude_questions:
        with open(args.exclude_questions) as f:
            for line in f:
                t = json.loads(line)["text"]
                seen_q.add(norm_q(t.split("\n\n")[0]))
        print("excluding", len(seen_q), "already-used questions")

    kept = load_omi(args.shards, want, args.cap_per_problem)

    rows = []
    for v in kept.values():
        rows.extend(v)
    if seen_q:
        before = len(rows)
        rows = [r for r in rows if norm_q(r["question"]) not in seen_q]
        print(f"dropped {before - len(rows)} rows whose question was already used")
    if not args.no_gsm8k_gold:
        rows.extend(load_gsm8k_train())
    rng.shuffle(rows)

    pool = fewshot_pool()
    n_fs = 0
    with open(args.out, "w") as f:
        for r in rows:
            system = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, 8)
                system = "\n\n".join(rng.sample(pool, k))
                n_fs += 1
            prompt = fmt.render_prompt(r["question"], system)
            completion = fmt.render_completion(r["target"])
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                # `text` is what contamination_check.py reads: the raw problem +
                # solution, without the boilerplate template that would otherwise
                # dominate every n-gram comparison.
                "text": r["question"] + "\n\n" + r["target"],
                "src": r["src"],
            }) + "\n")
    from collections import Counter
    print("wrote", len(rows), "rows to", args.out)
    print(Counter(r["src"] for r in rows))
    print("fewshot-prefixed rows:", n_fs)


if __name__ == "__main__":
    main()
