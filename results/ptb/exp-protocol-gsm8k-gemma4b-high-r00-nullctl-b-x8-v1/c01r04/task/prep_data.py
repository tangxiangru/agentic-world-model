#!/usr/bin/env python3
"""Build SFT data for GSM8K in the exact inspect-ai eval format."""
import argparse, glob, hashlib, json, random, re, os
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"

BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(s: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        j = m.end()
        depth = 1
        while j < len(s) and depth:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        out.append(s[m.end(): j - 1])
        i = j
    return "".join(out)


NUM_RE = re.compile(r"^-?\d{1,12}(,\d{3})*(\.\d+)?$")


def clean_number(a: str):
    a = a.strip().replace("$", "").replace(",", "").replace("%", "").strip()
    if not NUM_RE.match(a.replace(",", "")):
        return None
    try:
        f = float(a)
    except ValueError:
        return None
    if f == int(f) and abs(f) < 1e12:
        return str(int(f))
    return ("%g" % f)


BAD_MARKERS = ("```", "\\begin{", "http", "<<", "####", "\\text{Answer", "\\[")


def make_target(solution: str, answer: str):
    ans = clean_number(answer)
    if ans is None:
        return None
    sol = strip_boxed(solution).strip()
    if not sol or len(sol) < 20 or len(sol) > 3000:
        return None
    for b in BAD_MARKERS:
        if b in sol:
            return None
    # solution should reference the final answer somewhere near the end
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return f"{sol}\n\nANSWER: {ans}"


def sample_to_fewshot(q, reasoning, target):
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/sft.jsonl")
    ap.add_argument("--n-aug-gsm8k", type=int, default=120000)
    ap.add_argument("--n-gsm8k", type=int, default=30000)
    ap.add_argument("--n-math", type=int, default=15000)
    ap.add_argument("--max-per-problem", type=int, default=4)
    ap.add_argument("--skip-per-problem", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", type=int, default=0, help="which hash partition of problems to use")
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--exclude", default=None, help="file with one used problem per line (json string)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    excluded = set()
    if args.exclude:
        for line in open(args.exclude):
            excluded.add(json.loads(line))
        print("excluded problems", len(excluded))

    # ---- fewshot pool from GSM8K train (allowed: train split) ----
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    fewshot_pool = []
    train_questions = set()
    for r in gsm_train:
        q = r["question"]
        train_questions.add(q.strip())
        parts = r["answer"].split("####")
        tgt = parts[-1].strip()
        reasoning = "####".join(parts[:-1]).strip()
        fewshot_pool.append((q, reasoning, tgt))
    print("fewshot pool", len(fewshot_pool))

    # ---- collect solutions from OpenMathInstruct-2 ----
    files = sorted(glob.glob(OMI_GLOB))
    print("parquet files:", len(files))
    buckets = {"gsm8k": {}, "augmented_gsm8k": {}, "math": {}, "augmented_math": {}}
    caps = {"gsm8k": args.n_gsm8k, "augmented_gsm8k": args.n_aug_gsm8k,
            "math": args.n_math // 2, "augmented_math": args.n_math - args.n_math // 2}
    counts = {k: 0 for k in buckets}
    skipped = {}

    for f in files:
        df = pq.read_table(f, columns=["problem", "generated_solution", "expected_answer", "problem_source"]).to_pandas()
        for src, sub in df.groupby("problem_source"):
            if src not in buckets or counts[src] >= caps[src]:
                continue
            b = buckets[src]
            for problem, sol, ans in zip(sub["problem"], sub["generated_solution"], sub["expected_answer"]):
                if counts[src] >= caps[src]:
                    break
                if problem in excluded:
                    continue
                if args.nparts > 1 and src.startswith("augmented"):
                    h = int(hashlib.md5(problem.encode()).hexdigest()[:8], 16)
                    if h % args.nparts != args.part:
                        continue
                lst = b.setdefault(problem, [])
                nskip = skipped.setdefault(problem, 0)
                if len(lst) >= args.max_per_problem:
                    continue
                t = make_target(sol, ans)
                if t is None:
                    continue
                if any(t == x for x in lst):
                    continue
                if nskip < args.skip_per_problem:
                    skipped[problem] = nskip + 1
                    continue
                lst.append(t)
                counts[src] += 1
        print(os.path.basename(f), counts, flush=True)
        if all(counts[k] >= caps[k] for k in buckets):
            break

    records = []
    for src, b in buckets.items():
        for problem, sols in b.items():
            for s in sols:
                records.append({"question": problem, "target": s, "source": src})
    rng.shuffle(records)
    print("total records", len(records), {k: counts[k] for k in counts})

    # ---- render into chat format ----
    n_fs = 0
    with open(args.out, "w") as fh:
        for rec in records:
            msgs = []
            if rng.random() < args.fewshot_frac:
                k = rng.choices([2, 4, 10], weights=[0.4, 0.3, 0.3])[0]
                shots = rng.sample(fewshot_pool, k)
                msgs.append({"role": "system",
                             "content": "\n\n".join(sample_to_fewshot(*s) for s in shots)})
                n_fs += 1
            msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=rec["question"])})
            msgs.append({"role": "assistant", "content": rec["target"]})
            fh.write(json.dumps({"messages": msgs, "source": rec["source"]}) + "\n")
    print("wrote", args.out, "fewshot-prefixed:", n_fs)

    with open(args.out + ".problems", "w") as fh:
        for src_, b in buckets.items():
            for problem in b:
                fh.write(json.dumps(problem) + "\n")

    # decontamination input: question + solution text only
    with open(args.out + ".decon", "w") as fh:
        for rec in records:
            fh.write(json.dumps({"text": rec["question"] + "\n" + rec["target"]}) + "\n")


if __name__ == "__main__":
    main()
