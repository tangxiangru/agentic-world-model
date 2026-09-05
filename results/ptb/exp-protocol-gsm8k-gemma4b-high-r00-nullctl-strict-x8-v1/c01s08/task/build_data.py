#!/usr/bin/env python3
"""Build SFT data for GSM8K in the exact inspect_evals/gsm8k prompt format."""
from __future__ import annotations
import argparse, glob, json, random, re, os
import pyarrow.parquet as pq
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def strip_boxed(text: str) -> str:
    """Replace \\boxed{...} with its contents (brace-balanced)."""
    out = []
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j == -1:
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
            k += 1
        out.append(text[j + len("\\boxed{"): k - 1])
        i = k
    res = "".join(out)
    return res.replace("\\boxed", "")


NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def is_numeric(s: str) -> bool:
    return bool(NUM_RE.match(s.strip().replace(",", "").replace("$", "")))


def norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def clean_gt(answer: str):
    body, _, tgt = answer.partition("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return body, tgt.strip().replace(",", "")


def make_fewshot_block(samples) -> str:
    parts = []
    for q, reasoning, tgt in samples:
        parts.append(f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {tgt}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--n-aug-gsm", type=int, default=62000)
    ap.add_argument("--n-math", type=int, default=6000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gsm = load_dataset("openai/gsm8k", "main")["train"]
    # fewshot pool: raw reasoning WITH the <<>> annotations (matches inspect's format)
    fs_pool = []
    for r in gsm:
        body, _, tgt = r["answer"].partition("####")
        fs_pool.append((r["question"], body.strip(), tgt.strip()))

    records = []  # (question, solution_text, target)

    # --- 1. GSM8K train ground-truth solutions ---
    for r in gsm:
        body, tgt = clean_gt(r["answer"])
        if not body or not is_numeric(tgt):
            continue
        records.append((r["question"], body, norm_num(tgt), "gt"))
    print("gt:", len(records))

    # --- 2. OpenMathInstruct-2 ---
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet"))
    buckets = {"gsm8k": [], "augmented_gsm8k": [], "math": []}
    for f in files:
        t = pq.read_table(f).to_pandas()
        t = t[t.problem_source.isin(buckets.keys())]
        for row in t.itertuples(index=False):
            ans = str(row.expected_answer)
            if not is_numeric(ans):
                continue
            sol = strip_boxed(str(row.generated_solution)).strip()
            if not sol or len(sol) > 4000:
                continue
            buckets[row.problem_source].append((str(row.problem), sol, norm_num(ans)))
    for k, v in buckets.items():
        print(k, len(v))

    def dedup_cap(items, cap, limit=None):
        rng.shuffle(items)
        seen = {}
        out = []
        for q, s, a in items:
            c = seen.get(q, 0)
            if c >= cap:
                continue
            seen[q] = c + 1
            out.append((q, s, a))
            if limit and len(out) >= limit:
                break
        return out

    src = dedup_cap(buckets["gsm8k"], args.max_per_problem)
    records += [(q, s, a, "omi_gsm") for q, s, a in src]
    print("omi_gsm:", len(src))
    src = dedup_cap(buckets["augmented_gsm8k"], args.max_per_problem, args.n_aug_gsm)
    records += [(q, s, a, "omi_aug") for q, s, a in src]
    print("omi_aug:", len(src))
    src = dedup_cap(buckets["math"], 1, args.n_math)
    records += [(q, s, a, "omi_math") for q, s, a in src]
    print("omi_math:", len(src))

    rng.shuffle(records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_fs = 0
    with open(args.out, "w") as f:
        for q, sol, tgt, src_name in records:
            prompt = MATH_PROMPT_TEMPLATE.format(prompt=q.strip())
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 4, 5, 8, 10, 10, 10])
                demos = rng.sample(fs_pool, k)
                prompt = make_fewshot_block(demos) + "\n\n" + prompt
                n_fs += 1
            completion = f"{sol}\n\nANSWER: {tgt}"
            f.write(json.dumps({
                "prompt": prompt, "completion": completion,
                "source": src_name, "target": tgt,
                "text_for_decon": q.strip() + "\n" + sol + "\nANSWER: " + tgt,
            }) + "\n")
    print("total:", len(records), "with fewshot:", n_fs, "->", args.out)


if __name__ == "__main__":
    main()
