#!/usr/bin/env python3
"""Build the exp-04 training corpus: gsm8k train + all unique OpenMathInstruct-2
gsm8k problems + self-generated rejection-sampled solutions.

Same target shape and same renderer as build_data.py, so the two corpora are
interchangeable as far as the grader is concerned.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmt import ANSWER_MARKER, END_OF_TURN, render_prompt_fast  # noqa: E402
from eval_format import build_system_message, build_user_message  # noqa: E402
from build_data import clean_omi_solution, norm_answer  # noqa: E402

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(TASK_DIR, "data")


def signature(text: str) -> str:
    return "|".join(re.findall(r"\d+\.?\d*", text))


def load_holdout_questions() -> set[str]:
    return {
        json.loads(l)["question"]
        for l in open(os.path.join(DATA, "dev_gsm8k_trainholdout.jsonl"))
    }


def load_gsm8k(hold: set[str]) -> list[dict]:
    rows = []
    for line in open(os.path.join(DATA, "gsm8k_train_raw.jsonl")):
        r = json.loads(line)
        q = r["question"].strip()
        if q in hold:
            continue
        body, _, ans = r["answer"].rpartition("####")
        ans = norm_answer(ans)
        if ans is None:
            continue
        rows.append({"question": q, "target": f"{body.strip()}\n\n{ANSWER_MARKER}{ans}",
                     "source": "gsm8k_train"})
    return rows


def load_omi(hold: set[str], cap: int, seed: int) -> list[dict]:
    import pandas as pd

    paths = sorted(
        glob.glob(
            "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/*/data/train_1M-*.parquet"
        )
    )
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        frames.append(df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["problem"])
    rows = []
    for r in df.itertuples(index=False):
        q = str(r.problem).strip()
        if q in hold:
            continue
        ans = norm_answer(r.expected_answer)
        if ans is None:
            continue
        sol = clean_omi_solution(r.generated_solution)
        if not sol or "ANSWER" in sol:
            continue
        rows.append({"question": q, "target": f"{sol}\n\n{ANSWER_MARKER}{ans}",
                     "source": "omi"})
    random.Random(seed).shuffle(rows)
    return rows[:cap]


def load_rft(path: str, max_per_q: int, max_chars: int) -> tuple[list[dict], dict]:
    rows = []
    n_q = n_cov = 0
    if not os.path.exists(path):
        return rows, {"note": "no rft file", "questions": 0}
    for line in open(path):
        r = json.loads(line)
        n_q += 1
        cands = [s for s in r["samples"] if s["correct"] and s["finish"] == "stop"]
        cands.sort(key=lambda s: len(s["text"]))
        seen, keep = set(), []
        for s in cands:
            t = s["text"].strip()
            if t.count(ANSWER_MARKER) != 1 or len(t) > max_chars:
                continue
            sig = signature(t)
            if sig in seen:
                continue
            seen.add(sig)
            keep.append(t)
            if len(keep) >= max_per_q:
                break
        if keep:
            n_cov += 1
        for t in keep:
            rows.append({"question": r["question"], "target": t, "source": "rft_self"})
    return rows, {
        "questions": n_q,
        "questions_covered": n_cov,
        "coverage": round(n_cov / max(1, n_q), 4),
        "rows": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "sft_v2.jsonl"))
    ap.add_argument("--omi", type=int, default=65000)
    ap.add_argument("--rft", default=os.path.join(DATA, "rft_samples_gsm8k.jsonl"))
    ap.add_argument("--rft-max-per-q", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hold = load_holdout_questions()
    gsm = load_gsm8k(hold)
    omi = load_omi(hold, args.omi, args.seed)
    rft, rft_stats = load_rft(args.rft, args.rft_max_per_q, 2600)
    rft = [r for r in rft if r["question"] not in hold]

    rows = gsm + omi + rft
    rng.shuffle(rows)

    system = build_system_message()
    n_few = int(len(rows) * args.fewshot_frac)
    out = []
    for i, r in enumerate(rows):
        sysm = system if i < n_few else None
        out.append(
            {
                "prompt": render_prompt_fast(sysm, build_user_message(r["question"])),
                "completion": r["target"].strip() + END_OF_TURN,
                "source": r["source"],
                "fewshot": sysm is not None,
            }
        )
    rng.shuffle(out)
    with open(args.out, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")

    print(json.dumps({
        "gsm8k_train": len(gsm),
        "omi": len(omi),
        "rft": len(rft),
        "rft_stats": rft_stats,
        "total": len(out),
        "fewshot_rows": n_few,
        "out": args.out,
    }, indent=1))


if __name__ == "__main__":
    main()
