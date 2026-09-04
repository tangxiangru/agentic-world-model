"""Build the SFT mixture for GSM8K post-training of gemma-3-4b-pt.

Target format is dictated by the grader (inspect_evals/gsm8k + inspect_ai match
scorer, location="end", numeric=True): the LAST number in the completion must be
the gold answer. So every target ends with a single line "ANSWER: <int>" and
nothing after it.

Prompt format is the grader's MATH_PROMPT_TEMPLATE, rendered through
templates/gemma3.jinja (the exact template evaluate.py passes to vLLM).

Sources (all GSM8K *train*-derived or LLM-generated; the benchmark test split is
never read here):
  * openai/gsm8k train split, human-written solutions (calculator annotations stripped)
  * nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}
"""
import argparse
import json
import random
import re
from datasets import load_dataset

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def clean_solution(sol: str) -> str:
    sol = CALC.sub("", sol)
    sol = BOXED.sub(r"\1", sol)
    sol = sol.replace("\\[", "").replace("\\]", "")
    sol = re.sub(r"[ \t]+\n", "\n", sol)
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


def is_clean_int(s: str) -> bool:
    s = s.strip().replace(",", "")
    if s.startswith("-"):
        s = s[1:]
    return s.isdigit() and len(s) <= 12


def norm_int(s: str) -> str:
    return str(int(s.strip().replace(",", "")))


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--n-omi-gsm8k", type=int, default=25000, help="rows from OpenMathInstruct-2 problem_source=gsm8k")
    ap.add_argument("--n-omi-aug", type=int, default=30000, help="rows from OpenMathInstruct-2 problem_source=augmented_gsm8k")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    probe = {norm_q(json.loads(l)["question"]) for l in open("/home/ben/task/data/probe250.jsonl")}
    print(f"probe questions held out: {len(probe)}")

    rows = []

    # --- 1. original GSM8K train, human solutions -------------------------
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    n_gsm = 0
    for r in gsm:
        if norm_q(r["question"]) in probe:
            continue
        body, ans = r["answer"].split("####")
        if not is_clean_int(ans):
            continue
        sol = clean_solution(body)
        rows.append({
            "question": r["question"],
            "solution": sol,
            "answer": norm_int(ans),
            "source": "gsm8k_train_human",
        })
        n_gsm += 1
    print(f"gsm8k train human solutions: {n_gsm}")

    # --- 2. OpenMathInstruct-2 (gsm8k + augmented_gsm8k) ------------------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    omi = omi.filter(
        lambda b: [s in ("gsm8k", "augmented_gsm8k") for s in b["problem_source"]],
        batched=True, num_proc=16,
    )
    print(f"omi gsm8k-family rows: {len(omi)}")
    omi = omi.shuffle(seed=args.seed)

    per_problem = {}
    n_take = {"gsm8k": 0, "augmented_gsm8k": 0}
    want = {"gsm8k": args.n_omi_gsm8k, "augmented_gsm8k": args.n_omi_aug}
    for r in omi:
        src = r["problem_source"]
        if n_take[src] >= want[src]:
            if all(n_take[k] >= want[k] for k in want):
                break
            continue
        q = norm_q(r["problem"])
        if q in probe:
            continue
        if per_problem.get(q, 0) >= args.max_per_problem:
            continue
        ans = r["expected_answer"]
        if not is_clean_int(ans):
            continue
        sol = clean_solution(r["generated_solution"])
        if len(sol) < 20 or len(sol) > 4000:
            continue
        # the solution's own last number must already be the answer; otherwise the
        # appended ANSWER line would contradict the body
        nums = re.findall(r"-?\d[\d,]*", sol.replace("$", ""))
        if not nums or norm_int(nums[-1]) != norm_int(ans):
            continue
        per_problem[q] = per_problem.get(q, 0) + 1
        n_take[src] += 1
        rows.append({
            "question": r["problem"],
            "solution": sol,
            "answer": norm_int(ans),
            "source": f"omi_{src}",
        })
    print(f"omi taken: {n_take}")

    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            prompt = PROMPT_TEMPLATE.format(prompt=r["question"].strip())
            # the terminator the grading template stops on is part of the target
            target = f"{r['solution']}\n\nANSWER: {r['answer']}<end_of_turn>"
            f.write(json.dumps({
                "prompt": prompt,
                "completion": target,
                "question": r["question"],
                "answer": r["answer"],
                "source": r["source"],
            }) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
