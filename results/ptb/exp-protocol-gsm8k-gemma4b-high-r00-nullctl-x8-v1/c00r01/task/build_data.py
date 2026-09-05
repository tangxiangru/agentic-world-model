#!/usr/bin/env python3
"""Build SFT data for GSM8K in the exact format used by evaluate.py."""
from __future__ import annotations
import argparse, glob, json, os, random, re
import pandas as pd

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(text: str) -> str:
    """Remove \\boxed{...} wrappers, keeping their contents."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        j = m.end()
        depth = 1
        buf = []
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(text[j])
            j += 1
        out.append("".join(buf))
        i = j + 1
    return "".join(out)


def norm_num(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        f = float(s)
    except ValueError:
        return None
    if f == int(f):
        return str(int(f))
    return str(f)


def clean_solution(sol: str, answer: str) -> str | None:
    sol = strip_boxed(sol).strip()
    if "\\(" in sol or "\\[" in sol or "```" in sol:
        return None
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol + "\n\nANSWER: " + answer


def gsm8k_native(rec) -> tuple[str, str] | None:
    q = rec["question"].strip()
    a = rec["answer"]
    if "####" not in a:
        return None
    reasoning, target = a.split("####")
    target = norm_num(target)
    if target is None:
        return None
    return q, reasoning.strip() + "\n\nANSWER: " + target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-aug-gsm8k", type=int, default=60000)
    ap.add_argument("--n-gsm8k", type=int, default=30000)
    ap.add_argument("--n-math", type=int, default=12000)
    ap.add_argument("--fewshot-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    print("parquet shards:", len(files))

    buckets: dict[str, list] = {"augmented_gsm8k": [], "gsm8k": [], "math": []}
    per_problem: dict[str, int] = {}
    for f in files:
        df = pd.read_parquet(f, columns=["problem", "generated_solution",
                                         "expected_answer", "problem_source"])
        df = df[df.problem_source.isin(["augmented_gsm8k", "gsm8k", "math"])]
        for prob, sol, ans, src in df.itertuples(index=False):
            ans_n = norm_num(ans)
            if ans_n is None:
                continue
            if len(sol) > 2500 or len(prob) > 1500:
                continue
            key = prob[:200]
            if per_problem.get(key, 0) >= args.max_per_problem:
                continue
            c = clean_solution(sol, ans_n)
            if c is None:
                continue
            per_problem[key] = per_problem.get(key, 0) + 1
            buckets[src].append((prob.strip(), c))
        print("after", os.path.basename(f), {k: len(v) for k, v in buckets.items()}, flush=True)

    # native GSM8K train (matches the few-shot style shown at eval time)
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main")["train"]
    native = []
    fewshot_pool = []
    for rec in gsm:
        r = gsm8k_native(rec)
        if r:
            native.append(r)
            fewshot_pool.append((r[0], r[1]))
    print("native gsm8k train:", len(native))

    examples = []
    for name, n in [("augmented_gsm8k", args.n_aug_gsm8k), ("gsm8k", args.n_gsm8k),
                    ("math", args.n_math)]:
        pool = buckets[name]
        rng.shuffle(pool)
        examples += [(q, s, name) for q, s in pool[:n]]
    examples += [(q, s, "gsm8k_native") for q, s in native]
    rng.shuffle(examples)
    print("total examples:", len(examples))

    def make_fewshot_prefix(k: int, exclude_q: str) -> str:
        picks = []
        while len(picks) < k:
            q, s = fewshot_pool[rng.randrange(len(fewshot_pool))]
            if q == exclude_q:
                continue
            picks.append(f"{q}\n\nReasoning:\n{s}")
        return "\n\n".join(picks) + "\n\n"

    n_written = 0
    with open(args.out, "w") as fh:
        for q, s, src in examples:
            prefix = ""
            if rng.random() < args.fewshot_frac:
                prefix = make_fewshot_prefix(rng.choice([1, 2, 3, 4, 5, 8, 10]), q)
            prompt = ("<start_of_turn>user\n" + prefix
                      + MATH_PROMPT_TEMPLATE.format(prompt=q)
                      + "<end_of_turn>\n<start_of_turn>model\n")
            completion = s + "<end_of_turn>\n"
            fh.write(json.dumps({"prompt": prompt, "completion": completion,
                                 "source": src}) + "\n")
            n_written += 1
    print("wrote", n_written, "->", args.out)


if __name__ == "__main__":
    main()
