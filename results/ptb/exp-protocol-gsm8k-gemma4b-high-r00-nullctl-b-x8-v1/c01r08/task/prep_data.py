#!/usr/bin/env python3
"""Build SFT data for GSM8K from public math corpora (train splits only)."""
import argparse, glob, json, random, re, collections
import pyarrow.parquet as pq

PROMPT = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    "{q}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) '
    "where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n"
    "Reasoning:"
)

NUMRE = re.compile(r"^-?\d{1,12}(,\d{3})*(\.\d+)?$")


def unbox(text):
    """Replace every \\boxed{...} with its content."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len("\\boxed{")
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out.append(text[j + len("\\boxed{"): k])
        i = k + 1
    return "".join(out)


def clean_ans(a):
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    if a.endswith(".0"):
        a = a[:-2]
    if re.fullmatch(r"-?\d+\.\d*0+", a):
        a = a.rstrip("0").rstrip(".")
    return a


def is_num(a):
    return bool(NUMRE.match(a.strip()))


def make(q, sol, ans):
    sol = sol.strip()
    return {"question": q.strip(), "solution": sol + "\n\nANSWER: " + ans, "answer": ans}


def load_omi2(max_per_source, seed=0):
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    buckets = collections.defaultdict(list)
    per_problem = collections.Counter()
    caps = {"gsm8k": 4, "augmented_gsm8k": 2, "math": 2, "augmented_math": 1}
    for f in files:
        t = pq.read_table(f)
        cols = {c: t.column(c).to_pylist() for c in
                ["problem", "generated_solution", "expected_answer", "problem_source"]}
        for p, s, a, src in zip(cols["problem"], cols["generated_solution"],
                                cols["expected_answer"], cols["problem_source"]):
            if len(buckets[src]) >= max_per_source.get(src, 0):
                continue
            a = clean_ans(a)
            if not is_num(a):
                continue
            if "\\boxed{" not in s:
                continue
            if len(s) > 2600 or len(s) < 40 or len(p) > 1600:
                continue
            key = (src, p)
            if per_problem[key] >= caps.get(src, 1):
                continue
            body = unbox(s).strip()
            # drop degenerate/asy/table-heavy solutions
            if "[asy]" in body or "\\begin{tabular}" in body:
                continue
            per_problem[key] += 1
            buckets[src].append(make(p, body, a))
        if all(len(buckets[k]) >= v for k, v in max_per_source.items()):
            break
    return buckets


def load_gsm8k_train():
    from datasets import load_dataset
    d = load_dataset("openai/gsm8k", "main")["train"]
    out = []
    for r in d:
        q, a = r["question"], r["answer"]
        sol, tgt = a.split("####")
        sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        out.append(make(q, sol, clean_ans(tgt)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--n-gsm8k-omi", type=int, default=40000)
    ap.add_argument("--n-aug-gsm8k", type=int, default=90000)
    ap.add_argument("--n-math", type=int, default=12000)
    ap.add_argument("--n-aug-math", type=int, default=28000)
    ap.add_argument("--gsm8k-train-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    caps = {"gsm8k": args.n_gsm8k_omi, "augmented_gsm8k": args.n_aug_gsm8k,
            "math": args.n_math, "augmented_math": args.n_aug_math}
    b = load_omi2(caps)
    rows = []
    for k, v in b.items():
        print(k, len(v))
        rows += v
    gt = load_gsm8k_train()
    print("gsm8k_human", len(gt) * args.gsm8k_train_repeat)
    rows += gt * args.gsm8k_train_repeat

    # dedupe exact (question, solution)
    seen = set()
    ded = []
    for r in rows:
        k = (r["question"], r["solution"])
        if k in seen:
            continue
        seen.add(k)
        ded.append(r)
    random.shuffle(ded)
    print("total", len(ded))
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in ded:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
