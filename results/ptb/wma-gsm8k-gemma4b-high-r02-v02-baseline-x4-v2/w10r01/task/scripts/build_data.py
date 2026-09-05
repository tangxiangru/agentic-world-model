#!/usr/bin/env python3
"""Render the SFT corpus into {"prompt","target","src"} jsonl in grader format.

Sources are all derived from GSM8K *train* (or from public augmentations of it).
The benchmark test split is never read here.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from common import TASK_DIR, clean_gsm8k_reasoning, norm_answer, render_prompt, render_target

OUT = TASK_DIR / "data"
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def q_key(q: str) -> str:
    return re.sub(r"\W+", "", q.lower())[:220]


def emit(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path}: {len(rows)}")


def gsm8k_pool():
    out = []
    for line in (OUT / "train_pool.jsonl").open():
        r = json.loads(line)
        sol = clean_gsm8k_reasoning(r["answer"])
        ans = norm_answer(r["gold"])
        if not is_number(ans):
            continue
        out.append({"question": r["question"], "solution": sol, "answer": ans, "src": "gsm8k_train"})
    return out


def metamath(n_per_type):
    from datasets import load_dataset

    ds = load_dataset("meta-math/MetaMathQA", split="train")
    buckets = {}
    for r in ds:
        t = r["type"]
        if not t.startswith("GSM"):
            continue
        resp = r["response"]
        if "The answer is:" not in resp:
            continue
        body, ans = resp.rsplit("The answer is:", 1)
        ans = norm_answer(BOXED.sub(r"\1", ans).replace("$", "").strip())
        if not is_number(ans):
            continue
        body = BOXED.sub(r"\1", body).strip()
        # MetaMathQA GSM_AnsAug bodies carry gsm8k's own "#### N" line; leaving it
        # in would teach a second answer marker (pitfall: double_answer_format)
        body = body.split("####")[0].strip()
        if not body:
            continue
        buckets.setdefault(t, []).append(
            {"question": r["query"], "solution": body, "answer": ans, "src": f"metamath:{t}"}
        )
    out = []
    for t, rows in sorted(buckets.items()):
        random.Random(0).shuffle(rows)
        take = n_per_type.get(t, 0)
        out.extend(rows[:take])
        print(f"  metamath {t}: pool {len(rows)} -> take {min(take, len(rows))}")
    return out


def omi2(n_gsm):
    from datasets import load_dataset

    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    ds = ds.filter(lambda r: r["problem_source"] in ("gsm8k", "augmented_gsm8k"), num_proc=8)
    rows = []
    for r in ds:
        ans = norm_answer(str(r["expected_answer"]))
        if not is_number(ans):
            continue
        sol = BOXED.sub(r"\1", r["generated_solution"]).strip()
        # OMI2 solutions end with "The final answer is X." - drop that tail so the
        # target carries exactly one answer marker
        sol = re.sub(r"\n?The final answer is[^\n]*\.?\s*$", "", sol).strip()
        if not sol:
            continue
        rows.append({"question": r["problem"], "solution": sol, "answer": ans, "src": "omi2_gsm"})
    random.Random(0).shuffle(rows)
    return rows[:n_gsm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "sft_r1.jsonl"))
    ap.add_argument("--omi2", type=int, default=0)
    ap.add_argument("--mm-ansaug", type=int, default=0)
    ap.add_argument("--mm-rephrased", type=int, default=0)
    ap.add_argument("--mm-sv", type=int, default=0)
    ap.add_argument("--mm-fobar", type=int, default=0)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    args = ap.parse_args()

    raw = []
    base = gsm8k_pool()
    print(f"  gsm8k_train: {len(base)} x{args.gsm8k_repeat}")
    raw.extend(base * args.gsm8k_repeat)

    nper = {
        "GSM_AnsAug": args.mm_ansaug,
        "GSM_Rephrased": args.mm_rephrased,
        "GSM_SV": args.mm_sv,
        "GSM_FOBAR": args.mm_fobar,
    }
    if any(nper.values()):
        raw.extend(metamath(nper))
    if args.omi2:
        raw.extend(omi2(args.omi2))

    # dedup on (question, answer); keeps the repeat of gsm8k_train intact only
    # if the caller asked for it, so dedup runs before the repeat is applied
    seen = set()
    rows = []
    for r in raw:
        k = (q_key(r["question"]), r["answer"], r["src"].split(":")[0])
        if k in seen:
            continue
        seen.add(k)
        rows.append(
            {
                "prompt": render_prompt(r["question"]),
                "target": render_target(r["solution"], r["answer"]),
                "src": r["src"],
                "question": r["question"],
                "answer": r["answer"],
            }
        )
    random.Random(0).shuffle(rows)

    # invariant checks: one answer marker, one terminator, no second marker
    # (pitfalls double_answer_format / eos_mismatch)
    def ok(r):
        t = r["target"]
        return t.count("ANSWER: ") == 1 and t.endswith("<end_of_turn>") and "####" not in t

    good = [r for r in rows if ok(r)]
    bad = len(rows) - len(good)
    print(f"invariant check: dropped {bad} / {len(rows)} ({bad / max(1, len(rows)):.4%})")
    assert bad / max(1, len(rows)) < 0.01, "too many malformed targets; fix the cleaner"
    rows = good
    emit(rows, args.out)

    from collections import Counter

    print(Counter(r["src"] for r in rows))


if __name__ == "__main__":
    main()
