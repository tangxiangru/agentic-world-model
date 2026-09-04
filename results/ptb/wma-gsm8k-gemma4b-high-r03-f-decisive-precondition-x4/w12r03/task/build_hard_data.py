"""Failure-targeted training data from a sampling pass.

Two slices, both keyed on how the model actually performed:

  hard  - questions where none of the k samples was correct: take the 405B
          reference solutions from OpenMathInstruct-2 for exactly those problems
          (up to --ref-per-hard each). These are the problem shapes the model
          cannot currently solve at all.
  rft   - questions solved on some but not all samples: keep the model's own
          correct solutions (up to 2, de-duplicated by the sequence of numbers
          used). On-policy reinforcement of the paths that worked.

Questions solved on every sample contribute nothing: they are already learnt.
"""
import argparse
import json
import random
import re

from datasets import load_dataset

import harness_format as hf
from build_sft_data import clean_solution, is_clean_int, norm_int, norm_q


def sig(text):
    return tuple(re.findall(r"-?\d+\.?\d*", text))


def well_formed(c):
    c = c.strip()
    return (
        20 < len(c) <= 3000
        and c.count("ANSWER:") == 1
        and re.match(r"^ANSWER:\s*-?[\d,]+(\.\d+)?$", c.split("\n")[-1].strip())
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="/home/ben/task/analysis/rft_samples.jsonl")
    ap.add_argument("--out", default="/home/ben/task/data/hard_v1.jsonl")
    ap.add_argument("--ref-per-hard", type=int, default=4)
    ap.add_argument("--rft-per-question", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    samples = [json.loads(l) for l in open(args.samples)]
    hard, partial, easy = [], [], 0
    for r in samples:
        n_ok = sum(r["correct"])
        if n_ok == 0:
            hard.append(r)
        elif n_ok < len(r["correct"]):
            partial.append(r)
        else:
            easy += 1
    print(f"questions={len(samples)} never_solved={len(hard)} partly_solved={len(partial)} always_solved={easy}")

    rows = []

    # ---- slice 1: reference solutions for the never-solved questions --------
    want = {norm_q(r["question"]): r for r in hard}
    got = {}
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    omi = omi.filter(lambda b: [s in ("gsm8k", "augmented_gsm8k") for s in b["problem_source"]],
                     batched=True, num_proc=16)
    omi = omi.shuffle(seed=args.seed)
    for r in omi:
        q = norm_q(r["problem"])
        if q not in want or got.get(q, 0) >= args.ref_per_hard:
            continue
        ans = r["expected_answer"]
        if not is_clean_int(ans) or norm_int(ans) != norm_int(want[q]["gold"]):
            continue
        sol = clean_solution(r["generated_solution"])
        if len(sol) < 20 or len(sol) > 3000:
            continue
        nums = re.findall(r"-?\d[\d,]*", sol.replace("$", ""))
        if not nums or norm_int(nums[-1]) != norm_int(ans):
            continue
        got[q] = got.get(q, 0) + 1
        rows.append({"question": r["problem"],
                     "completion": f"{sol}\n\nANSWER: {norm_int(ans)}{hf.STOP_TOKEN}",
                     "answer": norm_int(ans), "source": "hard_reference"})
    print(f"hard slice: {len(rows)} reference solutions covering {len(got)}/{len(hard)} never-solved questions")

    # ---- slice 2: the model's own correct solutions on partly-solved -------
    n_rft = 0
    for r in partial:
        good = [c.strip() for c, ok in zip(r["completions"], r["correct"]) if ok and well_formed(c)]
        rng.shuffle(good)
        seen, picked = set(), []
        for c in good:
            s = sig(c)
            if s in seen:
                continue
            seen.add(s)
            picked.append(c)
            if len(picked) >= args.rft_per_question:
                break
        for c in picked:
            rows.append({"question": r["question"], "completion": c + hf.STOP_TOKEN,
                         "answer": r["gold"], "source": "rft_self"})
            n_rft += 1
    print(f"rft slice: {n_rft} self-generated correct solutions over {len(partial)} partly-solved questions")

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "prompt": hf.PROMPT_TEMPLATE.format(prompt=r["question"].strip()),
                "completion": r["completion"], "question": r["question"],
                "answer": r["answer"], "source": r["source"],
            }) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
