#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K.

Every row is {"prompt": <text fed to the model>, "completion": <target>} where
the pair, concatenated, is byte-identical to what templates/gemma3.jinja renders
for the graded conversation. The target ends with the one marker the grader
reads ("ANSWER: <n>") followed by the terminator the grading template stops on
(<end_of_turn>), so the eos-mismatch and double-marker pitfalls cannot bite.

Sources (all GSM8K *train* or independently authored; the test split is never
touched):
  - nvidia/OpenMathInstruct-2, problem_source in {gsm8k, augmented_gsm8k}
  - nvidia/OpenMathInstruct-2, problem_source == augmented_math (small share,
    for arithmetic/algebra breadth)
  - openai/gsm8k train split, original human solutions
"""
import argparse
import json
import random
import re
from collections import defaultdict

TASK = "/home/ben/task"

# Exactly inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE, .strip()ed as prompt_template does.
MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
CALC = re.compile(r"<<[^>]*>>")
NUMBERISH = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def render(user_content: str) -> str:
    """The prompt string templates/gemma3.jinja produces for a single user turn."""
    return f"{BOS}{SOT}user\n{user_content.strip()}{EOT}\n{SOT}model\n"


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    a = a.rstrip(".")
    if not NUMBERISH.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


def clean_solution(sol: str) -> str:
    """Unwrap \\boxed{} so the solution carries no second answer marker."""
    sol = BOXED.sub(r"\1", sol)
    sol = sol.replace("\\[", "").replace("\\]", "")
    return sol.strip()


def make_completion(solution: str, answer: str) -> str:
    return f"{solution}\n\nANSWER: {answer}{EOT}"


def sample_to_fewshot(question: str, reasoning: str, target: str) -> str:
    """Byte-identical to inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def load_gsm8k_train():
    rows = []
    with open(f"{TASK}/data/raw/gsm8k_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            parts = d["answer"].split("####")
            target = parts.pop().strip()
            reasoning = "####".join(parts).strip()
            rows.append({"question": d["question"].strip(),
                         "reasoning": reasoning,
                         "target": target})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{TASK}/data/sft_v1.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-gsm8k-omi", type=int, default=110000)
    ap.add_argument("--n-math-omi", type=int, default=12000)
    ap.add_argument("--gsm8k-orig-copies", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", default=f"{TASK}/data/probe250.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # Held-out probe problems (GSM8K train). Excluded by normalised exact match,
    # which catches the original problems in both sources. Caveat recorded on the
    # card: an `augmented_gsm8k` row is an LLM *rewrite* of a train problem, so a
    # rewrite of a held-out problem can survive this filter; the probe is a
    # ranking signal, not a clean generalisation measurement.
    def norm(q):
        return " ".join(q.split()).lower()

    held = set()
    if args.probe:
        held = {norm(json.loads(l)["question"]) for l in open(args.probe)}
    print(f"probe holdout: {len(held)} problems excluded from the corpus")

    gsm_train = [r for r in load_gsm8k_train() if norm(r["question"]) not in held]

    # ---- OpenMathInstruct-2 -------------------------------------------------
    per_problem = defaultdict(int)
    gsm_pool, math_pool = [], []
    with open(f"{TASK}/data/raw/omi2_1M.jsonl") as f:
        for line in f:
            d = json.loads(line)
            src = d["problem_source"]
            is_gsm = src in ("gsm8k", "augmented_gsm8k")
            if not is_gsm and src != "augmented_math":
                continue
            ans = clean_answer(d["expected_answer"] or "")
            if ans is None:
                continue
            sol = clean_solution(d["generated_solution"] or "")
            if not sol or "ANSWER:" in sol or "\\boxed" in sol:
                continue
            if len(sol) > 4000 or len(d["problem"]) > 2000:
                continue
            if " ".join(d["problem"].split()).lower() in held:
                continue
            key = (src, d["problem"])
            if per_problem[key] >= args.max_per_problem:
                continue
            per_problem[key] += 1
            rec = {"question": d["problem"].strip(), "solution": sol,
                   "answer": ans, "src": "omi2_gsm8k" if is_gsm else "omi2_math"}
            (gsm_pool if is_gsm else math_pool).append(rec)

    rng.shuffle(gsm_pool)
    rng.shuffle(math_pool)
    pool = gsm_pool[: args.n_gsm8k_omi] + math_pool[: args.n_math_omi]
    print(f"omi2: gsm8k pool {len(gsm_pool)} -> {min(len(gsm_pool), args.n_gsm8k_omi)}, "
          f"math pool {len(math_pool)} -> {min(len(math_pool), args.n_math_omi)}")

    # ---- original GSM8K train solutions ------------------------------------
    for _ in range(args.gsm8k_orig_copies):
        for r in gsm_train:
            sol = CALC.sub("", r["reasoning"]).strip()
            pool.append({"question": r["question"], "solution": sol,
                         "answer": r["target"].replace(",", ""), "src": "gsm8k_orig"})

    rng.shuffle(pool)

    # ---- render -------------------------------------------------------------
    n_fs = 0
    with open(args.out, "w") as out:
        for rec in pool:
            user = MATH_PROMPT_TEMPLATE.format(prompt=rec["question"])
            # A minority of rows carry a few-shot prefix in the same position and
            # format the grader puts its 10-shot system message, so the model is
            # not seeing that prefix for the first time at grading.
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 4, 6])
                demos = rng.sample(gsm_train, k)
                prefix = "\n\n".join(
                    sample_to_fewshot(d["question"], d["reasoning"], d["target"])
                    for d in demos)
                user = prefix + "\n\n" + user
                n_fs += 1
            out.write(json.dumps({
                "prompt": render(user),
                "completion": make_completion(rec["solution"], rec["answer"]),
                "src": rec["src"],
            }) + "\n")
    print(f"wrote {args.out}: {len(pool)} rows ({n_fs} with a few-shot prefix)")


if __name__ == "__main__":
    main()
