#!/usr/bin/env python3
"""sft_v2: same format as sft_v1, but drawn from OpenMathInstruct-2's train_5M
split, which holds 805,928 GSM8K-train-derived solutions over 81,069 unique
problems (~10 sampled solutions each). v1 used train_1M and got 60k rows over
roughly as many problems; v2 keeps up to --per-problem distinct solutions per
problem, so the model sees several correct chains for the same question.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/ben/task/scripts")
from build_sft_data import clean_gsm8k_reasoning, make_row, norm_answer, unbox  # noqa: E402
from eval_format import gsm8k_fewshot_system  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_v2.jsonl")
    ap.add_argument("--per-problem", type=int, default=2)
    ap.add_argument("--slice-from", type=int, default=0,
                    help="skip this many solutions per problem before taking --per-problem; "
                         "with the same seed this yields a corpus disjoint from an earlier build")
    ap.add_argument("--n-omi-math", type=int, default=15000)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--exact-fewshot-share", type=float, default=0.5,
                    help="share of the few-shot rows that use the grader's own "
                         "10-shot system message verbatim")
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from datasets import load_dataset

    holdout = set(json.load(open("/home/ben/task/data/train_holdout_ids.json")))
    tr = load_dataset("openai/gsm8k", "main", split="train")
    gsm_pool = []
    raw_pool = []   # same items, GSM8K's own <<a*b=c>> annotations kept, for demo prefixes
    for i, rec in enumerate(tr):
        if i in holdout:
            continue
        parts = rec["answer"].split("####")
        ans = norm_answer(parts[-1])
        if ans is None:
            continue
        raw = "####".join(parts[:-1]).strip()
        gsm_pool.append((rec["question"], clean_gsm8k_reasoning(raw), ans))
        raw_pool.append((rec["question"], raw, ans))
    print("A gsm8k train usable:", len(gsm_pool), flush=True)

    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_5M")
    gsm_src = {"gsm8k", "augmented_gsm8k"}
    math_src = {"math", "augmented_math"}
    by_problem = defaultdict(list)
    c_pool = []
    n_math_seen = 0
    for rec in omi:
        ps = rec["problem_source"]
        if ps not in gsm_src and ps not in math_src:
            continue
        ans = norm_answer(rec["expected_answer"])
        if ans is None:
            continue
        sol = rec["generated_solution"]
        if len(sol) > args.max_sol_chars or "\\boxed" not in sol:
            continue
        sol = unbox(sol)
        if sol is None or len(sol) < 20:
            continue
        if ps in gsm_src:
            by_problem[rec["problem"]].append((sol, ans))
        else:
            n_math_seen += 1
            # reservoir-free cheap subsample: keep roughly the first 4x target
            if len(c_pool) < args.n_omi_math * 4:
                c_pool.append((rec["problem"], sol, ans))
    print("B unique gsm8k-ish problems:", len(by_problem), "C math candidates:", len(c_pool), flush=True)

    b_items = []
    for prob, sols in by_problem.items():
        seen = set()
        uniq = []
        for s, a in sols:
            k = (len(s), s[:80], a)
            if k in seen:
                continue
            seen.add(k)
            uniq.append((s, a))
        rng.shuffle(uniq)
        for s, a in uniq[args.slice_from : args.slice_from + args.per_problem]:
            b_items.append((prob, s, a, "omi_gsm8k"))
    rng.shuffle(c_pool)
    c_items = [(q, s, a, "omi_math") for q, s, a in c_pool[: args.n_omi_math]]
    print("B rows:", len(b_items), "C rows:", len(c_items), flush=True)

    items = []
    for _ in range(args.gsm8k_repeat):
        items += [(q, r, a, "gsm8k_train") for q, r, a in gsm_pool]
    items += b_items + c_items
    rng.shuffle(items)

    # Demo prefixes keep GSM8K's <<a*b=c>> annotations, because the grader's own
    # 10-shot system message does; only the TARGETS are cleaned of them.
    demo_pool = list(raw_pool)
    rng.shuffle(demo_pool)
    demo_pool = demo_pool[:400]
    exact_sys = gsm8k_fewshot_system()

    def fewshot_sys(k: int) -> str:
        return "\n\n".join(
            f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}" for q, r, a in rng.sample(demo_pool, k)
        )

    n_fs = int(len(items) * args.fewshot_frac)
    rows = []
    n_exact = 0
    for i, (q, r, a, src) in enumerate(items):
        if i < n_fs:
            if rng.random() < args.exact_fewshot_share:
                sysmsg = exact_sys
                n_exact += 1
            else:
                sysmsg = fewshot_sys(rng.choice([1, 2, 3, 5, 8, 10]))
        else:
            sysmsg = None
        rows.append(make_row(q, r, a, sysmsg, src))
    print("fewshot rows:", n_fs, "of which the grader's exact 10-shot block:", n_exact)
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out, len(rows))
    print(Counter(r["meta"]["source"] for r in rows))

    dump = args.out.replace(".jsonl", "_decon_text.jsonl")
    with open(dump, "w") as f:
        for q, r, a, _ in items:
            f.write(json.dumps({"text": f"{q}\n{r}\n\nANSWER: {a}"}) + "\n")
    print("wrote", dump)


if __name__ == "__main__":
    main()
