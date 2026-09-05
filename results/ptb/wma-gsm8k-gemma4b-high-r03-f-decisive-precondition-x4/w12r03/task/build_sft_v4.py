"""Third-stage SFT data.

The integer-answer augmented_gsm8k *problems* are exhausted (a 5000-row probe of
the shuffled stream found 0 usable unseen ones: 72% are problems already used and
almost all of the rest have non-integer answers). So this stage widens the
mixture in the two directions that are still open:

  * more distinct 405B-written solutions to gsm8k-family problems already seen
    (different reasoning paths, solution text never used before), and
  * integer-answer rows from OpenMathInstruct-2's MATH family, which are new
    problems in a neighbouring domain.

Held-out probe250 questions are excluded from every source, unconditionally.
"""
import argparse
import json
import random
import re

from datasets import load_dataset

from build_sft_data import PROMPT_TEMPLATE, clean_solution, is_clean_int, norm_int, norm_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_v4.jsonl")
    ap.add_argument("--n-gsm-family", type=int, default=70000)
    ap.add_argument("--n-math-family", type=int, default=30000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    probe = {norm_q(json.loads(l)["question"]) for l in open("/home/ben/task/data/probe250.jsonl")}
    seen_sol = {hash(json.loads(l)["completion"])
                for f in ("sft_v1", "sft_v2") for l in open(f"/home/ben/task/data/{f}.jsonl")}
    print(f"probe {len(probe)}, solution texts already used {len(seen_sol)}")

    GSM = ("gsm8k", "augmented_gsm8k")
    MATH = ("math", "augmented_math")
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    omi = omi.filter(lambda b: [s in GSM + MATH for s in b["problem_source"]],
                     batched=True, num_proc=16)
    omi = omi.shuffle(seed=args.seed)

    rows, per_problem = [], {}
    n_gsm = n_math = 0
    for r in omi:
        fam = "gsm" if r["problem_source"] in GSM else "math"
        if fam == "gsm" and n_gsm >= args.n_gsm_family:
            if n_math >= args.n_math_family:
                break
            continue
        if fam == "math" and n_math >= args.n_math_family:
            continue
        q = norm_q(r["problem"])
        if q in probe:
            continue
        cap = args.max_per_problem if fam == "gsm" else 1
        if per_problem.get(q, 0) >= cap:
            continue
        ans = r["expected_answer"]
        if not is_clean_int(ans):
            continue
        sol = clean_solution(r["generated_solution"])
        if len(sol) < 20 or len(sol) > 3000:
            continue
        nums = re.findall(r"-?\d[\d,]*", sol.replace("$", ""))
        if not nums or norm_int(nums[-1]) != norm_int(ans):
            continue
        target = f"{sol}\n\nANSWER: {norm_int(ans)}<end_of_turn>"
        if hash(target) in seen_sol:
            continue
        per_problem[q] = per_problem.get(q, 0) + 1
        if fam == "gsm":
            n_gsm += 1
        else:
            n_math += 1
        rows.append({"question": r["problem"], "target": target,
                     "answer": norm_int(ans), "source": f"omi4_{r['problem_source']}"})
    print(f"taken: gsm-family {n_gsm}, math-family {n_math}")

    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "prompt": PROMPT_TEMPLATE.format(prompt=r["question"].strip()),
                "completion": r["target"], "question": r["question"],
                "answer": r["answer"], "source": r["source"],
            }) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
