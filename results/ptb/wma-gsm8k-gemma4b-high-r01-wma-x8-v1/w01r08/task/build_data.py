#!/usr/bin/env python3
"""Build the SFT dataset for GSM8K from OpenMathInstruct-2 (+ optional gsm8k train).

Output rows: {"prompt_user": str, "system": str|null, "completion": str}
The trainer renders them with templates/gemma3.jinja, exactly as the grader does.
"""
from __future__ import annotations
import argparse, glob, json, random, re, os

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
STOP_TOKEN = "<end_of_turn>"


def is_plain_number(s: str) -> bool:
    return bool(NUM_RE.match(s.strip().replace(",", "")))


def strip_boxed(sol: str) -> str:
    """Replace \\boxed{X} with X (balanced-brace aware), drop $..$ around it."""
    out = []
    i = 0
    while True:
        j = sol.find("\\boxed{", i)
        if j < 0:
            out.append(sol[i:])
            break
        out.append(sol[i:j])
        k = j + len("\\boxed{")
        depth = 1
        while k < len(sol) and depth:
            if sol[k] == "{":
                depth += 1
            elif sol[k] == "}":
                depth -= 1
            k += 1
        out.append(sol[j + len("\\boxed{"): k - 1])
        i = k
    s = "".join(out)
    s = s.replace("$", "")
    return s


def make_completion(solution: str, answer: str) -> str | None:
    s = strip_boxed(solution).strip()
    if not s:
        return None
    # the stop token is part of the target: the grader's template ends every
    # model turn with <end_of_turn> (id 106) and vLLM stops on it.
    return s + "\n\nANSWER: " + answer.strip() + STOP_TOKEN


def gsm8k_fewshot_pool():
    from datasets import load_dataset
    d = load_dataset("openai/gsm8k", "main")["train"]
    pool = []
    for r in d:
        q = r["question"]
        a = r["answer"].split("####")
        tgt = a.pop().strip()
        reasoning = "####".join(a).strip()
        pool.append(f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {tgt}")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm", type=int, default=52000)
    ap.add_argument("--n-math", type=int, default=8000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sol-chars", type=int, default=3500)
    ap.add_argument("--exclude", default=None,
                    help="comma-separated jsonls from previous builds; their problems are skipped")
    ap.add_argument("--exclude-completions", default=None,
                    help="comma-separated jsonls; their exact completions are skipped, so the "
                         "same problems can be reused with solutions the model has not seen")
    args = ap.parse_args()

    excluded = set()
    if args.exclude:
        for fn in args.exclude.split(","):
            for line in open(fn):
                excluded.add(json.loads(line)["prompt_user"])
        print(f"excluding {len(excluded)} problems already used")
    excluded_comp = set()
    if args.exclude_completions:
        for fn in args.exclude_completions.split(","):
            for line in open(fn):
                excluded_comp.add(json.loads(line)["completion"])
        print(f"excluding {len(excluded_comp)} completions already used")

    rng = random.Random(args.seed)
    files = sorted(glob.glob("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                             "snapshots/*/data/*.parquet"))
    assert files, "no OpenMathInstruct-2 parquet shards found"
    print(f"{len(files)} shards")

    import pyarrow.parquet as pq

    per_problem: dict[str, int] = {}
    gsm_rows, math_rows = [], []
    seen_sol = set()
    for f in files:
        if len(gsm_rows) >= args.n_gsm and len(math_rows) >= args.n_math:
            break
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20000):
            d = batch.to_pydict()
            for prob, sol, ans, src in zip(d["problem"], d["generated_solution"],
                                           d["expected_answer"], d["problem_source"]):
                is_gsm = src in ("gsm8k", "augmented_gsm8k")
                bucket = gsm_rows if is_gsm else math_rows
                cap = args.n_gsm if is_gsm else args.n_math
                if len(bucket) >= cap:
                    continue
                if not is_plain_number(ans):
                    continue
                if len(sol) > args.max_sol_chars or len(prob) > 2000:
                    continue
                if per_problem.get(prob, 0) >= args.max_per_problem:
                    continue
                if excluded and MATH_PROMPT_TEMPLATE.format(prompt=prob.strip()) in excluded:
                    continue
                comp = make_completion(sol, ans)
                if comp is None or "\\boxed" in comp:
                    continue
                # the grader reads the LAST number in the output: enforce it
                nums = re.findall(r"-?\d[\d,]*\.?\d*", comp[:-len("<end_of_turn>")])
                if not nums:
                    continue
                if nums[-1].replace(",", "").rstrip(".") != ans.strip().replace(",", ""):
                    continue
                if comp in excluded_comp:
                    continue
                h = hash(comp)
                if h in seen_sol:
                    continue
                seen_sol.add(h)
                per_problem[prob] = per_problem.get(prob, 0) + 1
                bucket.append({"problem": prob, "completion": comp, "answer": ans.strip(),
                               "source": src})
            if len(gsm_rows) >= args.n_gsm and len(math_rows) >= args.n_math:
                break
        print(f"  after {os.path.basename(f)}: gsm={len(gsm_rows)} math={len(math_rows)}",
              flush=True)

    rows = gsm_rows + math_rows
    rng.shuffle(rows)
    print(f"total {len(rows)} rows")

    # few-shot prefixes: a fraction of rows carry a system block, so the model is
    # robust to the grader's fixed 10-shot prompt.
    pool = gsm8k_fewshot_pool()
    exact10 = open("data/fewshot_system.txt").read()

    out = []
    for r in rows:
        system = None
        u = rng.random()
        if u < args.fewshot_frac * 0.25:
            system = exact10
        elif u < args.fewshot_frac:
            k = rng.choice([2, 3, 4])
            system = "\n\n".join(rng.sample(pool, k))
        out.append({"system": system,
                    "prompt_user": MATH_PROMPT_TEMPLATE.format(prompt=r["problem"].strip()),
                    "completion": r["completion"],
                    "answer": r["answer"],
                    "source": r["source"]})

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out, len(out))

    # a plain-text copy for the contamination checker (question + solution)
    ck = args.out.replace(".jsonl", "_forcheck.jsonl")
    with open(ck, "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["problem"] + "\n" + r["completion"].replace("<end_of_turn>", "")}) + "\n")
    print("wrote", ck)


if __name__ == "__main__":
    main()
