#!/usr/bin/env python3
"""Build SFT data in the grader's exact format.

Output jsonl rows: {"prompt": <rendered prompt string>, "completion": <target
string ending in <end_of_turn>>, "source": ..., "answer": ...}

Sources (none touch the gsm8k TEST split):
  * openai/gsm8k TRAIN split, minus a 500-item holdout kept as our dev set
  * nvidia/OpenMathInstruct-2, rows with problem_source in {gsm8k, augmented_gsm8k}
    (these are augmentations of the gsm8k TRAIN split)

Target shape mirrors inspect_evals' own few-shot rendering:
    "{reasoning}\n\nANSWER: {answer}" + "<end_of_turn>"
so that the grader's numeric end-match reads the ANSWER line.
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
from eval_format import build_user_message, build_system_message  # noqa: E402

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(TASK_DIR, "data")

HOLDOUT_N = 500
NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def norm_answer(a: str) -> str | None:
    a = str(a).strip().replace(",", "").replace("$", "")
    if not NUM_RE.match(a.replace(",", "")):
        return None
    # drop a trailing ".0" so the target reads like gsm8k's own answers
    if a.endswith(".0"):
        a = a[:-2]
    return a


def load_gsm8k_train() -> list[dict]:
    rows = []
    with open(os.path.join(DATA, "gsm8k_train_raw.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            body, _, ans = r["answer"].rpartition("####")
            ans = norm_answer(ans)
            if ans is None:
                continue
            rows.append(
                {
                    "question": r["question"].strip(),
                    "reasoning": body.strip(),
                    "answer": ans,
                    "source": "gsm8k_train",
                }
            )
    return rows


BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def clean_omi_solution(sol: str) -> str:
    # \boxed{40} -> 40 ; the ANSWER line we append is what the grader reads
    sol = BOXED_RE.sub(r"\1", sol)
    sol = sol.replace("\\[", "").replace("\\]", "")
    return sol.strip()


def load_openmathinstruct(max_rows: int, seed: int) -> list[dict]:
    import pandas as pd

    paths = sorted(
        glob.glob(
            os.path.expanduser(
                "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                "snapshots/*/data/train_1M-*.parquet"
            )
        )
    )
    if not paths:
        return []
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        frames.append(df[df.problem_source.isin(["gsm8k", "augmented_gsm8k"])])
    df = pd.concat(frames, ignore_index=True)
    # one solution per problem
    df = df.drop_duplicates(subset=["problem"]).reset_index(drop=True)
    rows = []
    for r in df.itertuples(index=False):
        ans = norm_answer(r.expected_answer)
        if ans is None:
            continue
        sol = clean_omi_solution(r.generated_solution)
        if not sol or ANSWER_MARKER.strip() in sol:
            continue
        rows.append(
            {
                "question": str(r.problem).strip(),
                "reasoning": sol,
                "answer": ans,
                "source": "openmathinstruct2_gsm8k",
            }
        )
    random.Random(seed).shuffle(rows)
    return rows[:max_rows]


def to_example(row: dict, system: str | None) -> dict:
    target = f"{row['reasoning'].strip()}\n\n{ANSWER_MARKER}{row['answer']}"
    return {
        "prompt": render_prompt_fast(system, build_user_message(row["question"])),
        "completion": target + END_OF_TURN,
        "answer": row["answer"],
        "source": row["source"],
        "fewshot": system is not None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "sft_v1.jsonl"))
    ap.add_argument("--omi", type=int, default=30000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm = load_gsm8k_train()
    rng.Random = None  # noqa: B010  (guard against accidental reuse)
    rng = random.Random(args.seed)
    idx = list(range(len(gsm)))
    rng.shuffle(idx)
    holdout_idx = set(idx[:HOLDOUT_N])
    holdout = [gsm[i] for i in sorted(holdout_idx)]
    train_gsm = [gsm[i] for i in range(len(gsm)) if i not in holdout_idx]

    with open(os.path.join(DATA, "dev_gsm8k_trainholdout.jsonl"), "w") as f:
        for i, r in enumerate(holdout):
            f.write(
                json.dumps(
                    {"id": f"gsm8ktrain-{i}", "question": r["question"], "gold": r["answer"]}
                )
                + "\n"
            )

    omi = load_openmathinstruct(args.omi, args.seed)
    # never train on a question that is in our own dev holdout
    hold_q = {r["question"] for r in holdout}
    omi = [r for r in omi if r["question"] not in hold_q]

    rows = train_gsm + omi
    rng.shuffle(rows)

    system = build_system_message()
    n_few = int(len(rows) * args.fewshot_frac)
    out = []
    for i, r in enumerate(rows):
        out.append(to_example(r, system if i < n_few else None))
    rng.shuffle(out)

    with open(args.out, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")

    print(f"gsm8k_train kept   : {len(train_gsm)}")
    print(f"holdout (dev)      : {len(holdout)} -> data/dev_gsm8k_trainholdout.jsonl")
    print(f"openmathinstruct2  : {len(omi)}")
    print(f"total rows         : {len(out)}  ({n_few} with the eval's 10-shot prefix)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
