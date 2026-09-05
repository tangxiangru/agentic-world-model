"""Second-stage SFT data: GSM8K-family rows that exp-02 has NOT already seen.

Excludes every question in sft_v1.jsonl (exp-02's data), the held-out probe, and
the rejection-sampling pool, so a continued run is fresh-data scaling rather
than a second epoch.
"""
import argparse
import json
import re

from datasets import load_dataset

from build_sft_data import PROMPT_TEMPLATE, clean_solution, is_clean_int, norm_int, norm_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_v2.jsonl")
    ap.add_argument("--n-aug", type=int, default=62000)
    ap.add_argument("--n-gsm-extra", type=int, default=14000,
                    help="further solutions to original gsm8k train problems")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    probe = {norm_q(json.loads(l)["question"]) for l in open("/home/ben/task/data/probe250.jsonl")}
    excl = set()
    for f, key in [("data/probe250.jsonl", "question"), ("data/sft_v1.jsonl", "question"),
                   ("data/rft_pool.jsonl", "question")]:
        n0 = len(excl)
        for line in open(f"/home/ben/task/{f}"):
            excl.add(norm_q(json.loads(line)[key]))
        print(f"{f}: +{len(excl) - n0} excluded questions")

    # questions whose *problems* may repeat (original gsm8k train) are allowed for
    # the extra-solution slice, so track those separately
    gsm_train = {norm_q(r["question"]) for r in load_dataset("openai/gsm8k", "main", split="train")}
    # exact solution texts already used in sft_v1, so extra solutions are new ones
    seen_sol = {hash(json.loads(l)["completion"]) for l in open("/home/ben/task/data/sft_v1.jsonl")}

    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    omi = omi.filter(lambda b: [s in ("gsm8k", "augmented_gsm8k") for s in b["problem_source"]],
                     batched=True, num_proc=16)
    omi = omi.shuffle(seed=args.seed)

    rows, per_problem = [], {}
    n_take = {"gsm8k": 0, "augmented_gsm8k": 0}
    want = {"gsm8k": args.n_gsm_extra, "augmented_gsm8k": args.n_aug}
    for r in omi:
        src = r["problem_source"]
        if n_take[src] >= want[src]:
            if all(n_take[k] >= want[k] for k in want):
                break
            continue
        q = norm_q(r["problem"])
        # the held-out probe is excluded from every source, unconditionally
        if q in probe:
            continue
        # gsm8k-source rows are deliberately extra solutions to original train
        # problems we already use (and which the rft_pool also draws on - RFT
        # conventionally samples on its training questions); augmented rows must
        # be problems exp-02 has never seen
        if src == "augmented_gsm8k" and q in excl:
            continue
        if src == "gsm8k" and q not in gsm_train:
            continue
        if per_problem.get(q, 0) >= args.max_per_problem:
            continue
        ans = r["expected_answer"]
        if not is_clean_int(ans):
            continue
        sol = clean_solution(r["generated_solution"])
        if len(sol) < 20 or len(sol) > 4000:
            continue
        nums = re.findall(r"-?\d[\d,]*", sol.replace("$", ""))
        if not nums or norm_int(nums[-1]) != norm_int(ans):
            continue
        target = f"{sol}\n\nANSWER: {norm_int(ans)}<end_of_turn>"
        if hash(target) in seen_sol:
            continue
        per_problem[q] = per_problem.get(q, 0) + 1
        n_take[src] += 1
        rows.append({"question": r["problem"], "solution": sol,
                     "answer": norm_int(ans), "source": f"omi2_{src}"})
    print("taken:", n_take)

    import random
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "prompt": PROMPT_TEMPLATE.format(prompt=r["question"].strip()),
                "completion": f"{r['solution']}\n\nANSWER: {r['answer']}<end_of_turn>",
                "question": r["question"], "answer": r["answer"], "source": r["source"],
            }) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
