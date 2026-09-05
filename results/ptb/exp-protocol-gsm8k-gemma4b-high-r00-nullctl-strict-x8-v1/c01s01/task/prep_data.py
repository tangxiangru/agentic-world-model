#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 (gsm8k + augmented_gsm8k) and GSM8K train."""
import argparse, glob, json, random, re, os
import pyarrow.parquet as pq
from datasets import load_dataset

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOX = re.compile(r"\\boxed\{")


def strip_boxed(s: str) -> str:
    """Replace \boxed{X} with X (handles nested braces)."""
    while True:
        m = BOX.search(s)
        if not m:
            return s
        start = m.end()  # index just after '{'
        depth = 1
        i = start
        while i < len(s) and depth:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        if depth:
            return s[: m.start()] + s[start:]
        s = s[: m.start()] + s[start : i - 1] + s[i:]


LATEX_SUBS = [
    (re.compile(r"\\\[|\\\]"), ""),
    (re.compile(r"\\\(|\\\)"), ""),
    (re.compile(r"\\text\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\times"), "*"),
    (re.compile(r"\\cdot"), "*"),
    (re.compile(r"\\div"), "/"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\\$"), "$"),
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"[ \t]+\n"), "\n"),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def clean_solution(s: str) -> str:
    s = strip_boxed(s)
    for pat, rep in LATEX_SUBS:
        s = pat.sub(rep, s)
    return s.strip()


INT_RE = re.compile(r"^-?\d+$")


def is_int_answer(a: str) -> bool:
    a = a.strip().replace(",", "")
    return bool(INT_RE.match(a))


def clean_gsm8k_ref(ans: str) -> tuple[str, str]:
    """GSM8K reference answer -> (reasoning, final)."""
    body, final = ans.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return body, final.strip().replace(",", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/sft_data.jsonl")
    ap.add_argument("--n-omi", type=int, default=90000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    # few-shot pool rendered in the exact eval style
    fewshot_pool = []
    for r in gsm_train:
        body, final = clean_gsm8k_ref(r["answer"])
        raw_body = r["answer"].split("####")[0].strip()
        fewshot_pool.append(f"{r['question']}\n\nReasoning:\n{raw_body}\n\nANSWER: {final}")

    records = []

    # ---- OpenMathInstruct-2 gsm8k subsets ----
    files = sorted(glob.glob(os.path.expanduser("~/hf_cache/**/train_1M-*.parquet"), recursive=True))
    per_problem = {}
    pool = []
    for f in files:
        t = pq.read_table(f).to_pandas()
        t = t[t.problem_source.isin(["gsm8k", "augmented_gsm8k"])]
        for prob, sol, ans, src in zip(t.problem, t.generated_solution, t.expected_answer, t.problem_source):
            if not is_int_answer(ans):
                continue
            k = per_problem.get(prob, 0)
            if k >= args.max_per_problem:
                continue
            sol = clean_solution(sol)
            if len(sol) < 40 or len(sol) > 2500:
                continue
            if "\\boxed" in sol or "\\begin" in sol:
                continue
            per_problem[prob] = k + 1
            pool.append({"question": prob.strip(), "solution": sol,
                         "answer": ans.strip().replace(",", ""), "src": src})
    print(f"OMI2 pool: {len(pool)} from {len(per_problem)} problems")
    rng.shuffle(pool)
    records.extend(pool[: args.n_omi])

    # ---- original GSM8K train reference solutions ----
    for r in gsm_train:
        body, final = clean_gsm8k_ref(r["answer"])
        records.append({"question": r["question"].strip(), "solution": body,
                        "answer": final, "src": "gsm8k_ref"})

    rng.shuffle(records)

    n_fs = int(len(records) * args.fewshot_frac)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for i, r in enumerate(records):
            user = PROMPT_TEMPLATE.format(prompt=r["question"])
            if i < n_fs:
                k = rng.randint(1, 10)
                shots = rng.sample(fewshot_pool, k)
                user = "\n\n".join(shots) + "\n\n" + user
            completion = r["solution"] + f"\n\nANSWER: {r['answer']}"
            fh.write(json.dumps({"prompt": user, "completion": completion,
                                 "src": r["src"]}) + "\n")
    print(f"wrote {len(records)} -> {args.out} ({n_fs} with fewshot prefix)")


if __name__ == "__main__":
    main()
