#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K, rendered through the grader's own surface.

Sources (all GSM8K *train* derived or public math corpora; the GSM8K test split
is never read here):
  A  nvidia/OpenMathInstruct-2 train_1M, rows with problem_source in
     {gsm8k, augmented_gsm8k}                       -> bulk
  B  openai/gsm8k main/train, human reference solutions
  C  nvidia/OpenMathInstruct-2 train_1M, rows with problem_source in
     {math, augmented_math} whose answer is a plain number -> small generality mix

Every row is emitted as {"prompt": <rendered chat prompt>, "completion": <text>}
where completion ends with a single 'ANSWER: <number>' line followed by the
grader's stop token <end_of_turn> (kept in the file so preflight can verify it).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys

sys.path.insert(0, "/home/ben/task/scripts")
import fmt  # noqa: E402

OMI_GLOB = ("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
            "snapshots/469216e3f46f4dacf476b382e192485ea51a143e/data/train_1M-*.parquet")
GSM_TRAIN = ("/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/"
             "740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet")

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def clean_solution(sol: str, answer: str) -> str | None:
    """Strip \\boxed{} and force exactly one trailing 'ANSWER: n' line."""
    s = sol.strip()
    # unwrap \boxed{...} (non-nested; rows with nested braces are dropped)
    if "\\boxed" in s:
        s2 = BOXED_RE.sub(lambda m: m.group(1), s)
        if "\\boxed" in s2:
            return None
        s = s2
    if "ANSWER:" in s:
        return None
    # OpenMathInstruct solutions end with a sentence naming the answer; keep it,
    # then add the marker line the grader reads.
    s = s.rstrip()
    if not s:
        return None
    return s + "\n\nANSWER: " + answer + fmt.STOP_TOKEN


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    if a.startswith("."):
        return None
    # drop trailing .0 so the string matches gsm8k's integer gold style
    if "." in a:
        f = float(a)
        if f.is_integer():
            a = str(int(f))
    return a


def load_omi():
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(OMI_GLOB)):
        t = pq.read_table(f)
        rows.extend(t.to_pylist())
    return rows


def load_gsm_train():
    import pyarrow.parquet as pq
    return pq.read_table(GSM_TRAIN).to_pylist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm-aug", type=int, default=80000)
    ap.add_argument("--n-math", type=int, default=8000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-completion-chars", type=int, default=2600)
    ap.add_argument("--skip-gsm-aug", type=int, default=0,
                    help="skip the first N rows of the (seeded) gsm pool - lets a later "
                         "round train on rows an earlier round never saw")
    ap.add_argument("--skip-math", type=int, default=0)
    ap.add_argument("--no-human-gsm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-fewshot-small", type=float, default=0.08)
    ap.add_argument("--p-fewshot-official", type=float, default=0.04)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(fmt.BASE_SNAPSHOT)
    official_sys = open("/home/ben/task/data/official_fewshot_system.txt").read()

    gsm_train = load_gsm_train()
    # pool of shots for the small-k few-shot variants (gsm8k TRAIN only)
    shot_pool = []
    for r in gsm_train:
        parts = r["answer"].split("####")
        tgt = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        shot_pool.append(fmt.fewshot_block(r["question"], reasoning, tgt))

    pool_gsm, pool_math = [], []
    seen_per_problem: dict[str, int] = {}
    dropped = {"answer": 0, "boxed": 0, "long": 0, "cap": 0}

    for r in load_omi():
        src = r["problem_source"]
        ans = norm_answer(r["expected_answer"] or "")
        if ans is None:
            dropped["answer"] += 1
            continue
        sol = clean_solution(r["generated_solution"] or "", ans)
        if sol is None:
            dropped["boxed"] += 1
            continue
        if len(sol) > args.max_completion_chars or len(r["problem"]) > 2000:
            dropped["long"] += 1
            continue
        key = r["problem"]
        n = seen_per_problem.get(key, 0)
        if n >= args.max_per_problem:
            dropped["cap"] += 1
            continue
        seen_per_problem[key] = n + 1
        rec = {"question": r["problem"], "completion": sol, "src": src}
        (pool_gsm if src in ("gsm8k", "augmented_gsm8k") else pool_math).append(rec)

    rng.shuffle(pool_gsm)
    rng.shuffle(pool_math)
    print(f"pool_gsm={len(pool_gsm)} pool_math={len(pool_math)} dropped={dropped}", flush=True)

    picked = (pool_gsm[args.skip_gsm_aug: args.skip_gsm_aug + args.n_gsm_aug]
              + pool_math[args.skip_math: args.skip_math + args.n_math])

    # source B: human GSM8K train references, in the grader's own fewshot style
    for r in ([] if args.no_human_gsm else gsm_train):
        parts = r["answer"].split("####")
        tgt = norm_answer(parts.pop().strip())
        if tgt is None:
            continue
        reasoning = "####".join(parts).strip()
        picked.append({"question": r["question"],
                       "completion": reasoning + "\n\nANSWER: " + tgt + fmt.STOP_TOKEN,
                       "src": "gsm8k_train_human"})

    rng.shuffle(picked)

    n_small = n_off = 0
    with open(args.out, "w") as f:
        for rec in picked:
            u = rng.random()
            if u < args.p_fewshot_official:
                system = official_sys
                n_off += 1
            elif u < args.p_fewshot_official + args.p_fewshot_small:
                k = rng.randint(1, 4)
                system = "\n\n".join(rng.sample(shot_pool, k))
                n_small += 1
            else:
                system = None
            prompt, _ = fmt.render(tok, system, rec["question"], None)
            f.write(json.dumps({"prompt": prompt,
                                "completion": rec["completion"],
                                "src": rec["src"]}) + "\n")
    print(f"wrote {len(picked)} rows to {args.out}; fewshot_official={n_off} fewshot_small={n_small}")

    # plain-text view for the contamination checker (question + answer fields)
    with open(args.out.replace(".jsonl", "_decon.jsonl"), "w") as f:
        for rec in picked:
            f.write(json.dumps({"question": rec["question"], "answer": rec["completion"]}) + "\n")


if __name__ == "__main__":
    main()
