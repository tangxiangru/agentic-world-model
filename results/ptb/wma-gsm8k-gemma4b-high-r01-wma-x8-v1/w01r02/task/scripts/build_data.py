#!/usr/bin/env python3
"""Build the SFT file for the gsm8k post-training run.

Sources (both derived from the gsm8k TRAIN split only, never the test split):
  * nvidia/OpenMathInstruct-2, train_1M, problem_source in {gsm8k, augmented_gsm8k}
  * openai/gsm8k train pool (7173 rows; probe300 and nothing else removed)

Every row is checked with fmt.grade(): the string the grader would extract from
the target must equal the gold answer. That is the guard against the
double_answer_format pitfall - the last number in the target is the answer.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

BOXED = re.compile(r"\\boxed\{")
CALC = re.compile(r"<<[^>]*>>")
INT_ANS = re.compile(r"^-?\d+$")


def unbox(s: str) -> str:
    """Replace every \\boxed{X} with X (brace-balanced)."""
    while True:
        m = BOXED.search(s)
        if not m:
            return s
        i = m.end()
        depth = 1
        j = i
        while j < len(s) and depth:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        if depth:
            return s[: m.start()] + s[i:]
        s = s[: m.start()] + s[i : j - 1] + s[j:]


def clean_solution(sol: str) -> str:
    sol = unbox(sol)
    sol = CALC.sub("", sol)
    sol = re.sub(r"[ \t]+\n", "\n", sol)
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--out", type=str, default=str(DATA / "sft_v1.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=str, default="train_1M")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probe_q = set(json.load(open(DATA / "probe_questions.json")))
    fewshot_sys = open(DATA / "fewshot_system.txt").read()
    # the 10 exemplars, split so shorter k-shot prefixes can be built
    shots = fewshot_sys.split("\n\n" + "")
    shots = [s for s in re.split(r"\n\n(?=[^\n]*\n\nReasoning:)", fewshot_sys)]

    stats = defaultdict(int)
    per_problem: dict[str, int] = defaultdict(int)
    rows: list[dict] = []

    # ---- OpenMathInstruct-2 -------------------------------------------------
    pat = f"hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/{args.split}-*.parquet"
    files = sorted(glob.glob(str(Path.home() / pat)))
    assert files, "OpenMathInstruct-2 shards not downloaded"
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"])
        d = t.to_pydict()
        for prob, sol, ans, src in zip(d["problem"], d["generated_solution"], d["expected_answer"], d["problem_source"]):
            stats["omi_seen"] += 1
            if src not in ("gsm8k", "augmented_gsm8k"):
                continue
            stats["omi_gsm_src"] += 1
            if prob in probe_q:
                stats["drop_probe"] += 1
                continue
            ans = ans.strip()
            if not INT_ANS.match(ans):
                stats["drop_nonint"] += 1
                continue
            if per_problem[prob] >= args.max_per_problem:
                stats["drop_cap"] += 1
                continue
            sol = clean_solution(sol)
            if not sol or len(sol) > args.max_chars:
                stats["drop_len"] += 1
                continue
            target = fmt.render_target(sol, ans)
            # the grader never sees the stop token (vLLM strips it), so grade without it
            if not fmt.grade(target[: -len(fmt.END)], ans):
                stats["drop_grade"] += 1
                continue
            per_problem[prob] += 1
            rows.append({"question": prob, "target_body": sol, "answer": ans, "src": src})
            stats["omi_kept"] += 1

    # ---- original gsm8k train pool -----------------------------------------
    pool = [json.loads(l) for l in open(DATA / "gsm8k_train_pool.jsonl")]
    for r in pool:
        body, _, _ = r["answer"].rpartition("####")
        body = clean_solution(body)
        ans = r["gold"]
        if not INT_ANS.match(ans):
            stats["drop_nonint_gsm"] += 1
            continue
        target = fmt.render_target(body, ans)
        if not fmt.grade(target[: -len(fmt.END)], ans):
            stats["drop_grade_gsm"] += 1
            continue
        for _ in range(args.gsm8k_repeat):
            rows.append({"question": r["question"], "target_body": body, "answer": ans, "src": "gsm8k_human"})
            stats["gsm_kept"] += 1

    # ---- attach few-shot prefixes to a fraction of rows ---------------------
    rng.shuffle(rows)
    n_fs = int(len(rows) * args.fewshot_frac)
    k_choices = [1, 2, 3, 4, 10]
    k_weights = [0.30, 0.25, 0.15, 0.15, 0.15]
    out = []
    for i, r in enumerate(rows):
        if i < n_fs:
            k = rng.choices(k_choices, k_weights)[0]
            sel = rng.sample(shots, k) if k < len(shots) else shots
            system = "\n\n".join(sel) if k < len(shots) else fewshot_sys
        else:
            system = None
        out.append(
            {
                "prompt": fmt.render_prompt(r["question"], system),
                "completion": fmt.render_target(r["target_body"], r["answer"]),
                "question": r["question"],
                "answer": r["answer"],
                "src": r["src"],
                "nshot": (k if i < n_fs else 0),
            }
        )
    rng.shuffle(out)

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(json.dumps(dict(stats), indent=2))
    print(f"n_shots parsed from fewshot_system.txt: {len(shots)}")
    print(f"wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
