#!/usr/bin/env python3
"""Build the SFT dataset from OpenMathInstruct-2 (GSM8K-derived subsets) + GSM8K train."""
import argparse, glob, json, random, re, os
import pandas as pd

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def strip_boxed(text: str) -> str:
    """Replace \\boxed{...} (and \\boxed ...) with plain contents."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("\\boxed", i)
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len("\\boxed")
        while k < n and text[k] == " ":
            k += 1
        if k < n and text[k] == "{":
            depth = 0
            start = k + 1
            while k < n:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            out.append(text[start:k])
            i = k + 1
        else:
            # \boxed followed by a bare token
            m = re.match(r"[^\s$]*", text[k:])
            out.append(m.group(0))
            i = k + m.end()
    return "".join(out)


LATEX_CLEAN = [
    (re.compile(r"\\text\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\dfrac"), r"\\frac"),
    (re.compile(r"\\!"), ""),
    (re.compile(r"\\,"), " "),
    (re.compile(r"\\\$"), "$"),
]


def clean_solution(sol: str) -> str:
    sol = strip_boxed(sol)
    for pat, rep in LATEX_CLEAN:
        sol = pat.sub(rep, sol)
    sol = re.sub(r"[ \t]+\n", "\n", sol)
    sol = re.sub(r"\n{3,}", "\n\n", sol)
    return sol.strip()


def is_int_answer(a: str) -> bool:
    a = a.strip().replace(",", "")
    if a.startswith("-"):
        a = a[1:]
    return a.isdigit()


def norm_int(a: str) -> str:
    return str(int(a.strip().replace(",", "")))


def build_response(sol: str, ans: str) -> str:
    sol = clean_solution(sol)
    return f"{sol}\n\nANSWER: {ans}"


def sample_to_fewshot(q, reasoning, target):
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--n-total", type=int, default=120000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.30)
    ap.add_argument("--math-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ---- fewshot pool: GSM8K train (official terse solutions, matches eval style)
    from datasets import load_dataset
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    fs_pool = []
    for r in gsm_train:
        parts = r["answer"].split("####")
        tgt = parts[-1].strip()
        reasoning = "####".join(parts[:-1]).strip()
        fs_pool.append((r["question"], reasoning, tgt))
    print("fewshot pool", len(fs_pool))

    # ---- main pool from OpenMathInstruct-2
    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    print("shards", len(files))
    per_problem = {}
    gsm_rows, math_rows = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["problem", "generated_solution",
                                         "expected_answer", "problem_source"])
        for prob, sol, ans, src in df.itertuples(index=False):
            if not is_int_answer(ans):
                continue
            if len(sol) > 3500 or len(sol) < 40:
                continue
            if src in ("gsm8k", "augmented_gsm8k"):
                bucket = gsm_rows
            elif src in ("math", "augmented_math"):
                bucket = math_rows
            else:
                continue
            c = per_problem.get(prob, 0)
            if c >= args.max_per_problem:
                continue
            per_problem[prob] = c + 1
            bucket.append((prob, sol, norm_int(ans), src))
        print(f, len(gsm_rows), len(math_rows), flush=True)

    print("gsm rows", len(gsm_rows), "math rows", len(math_rows))
    rng.shuffle(gsm_rows)
    rng.shuffle(math_rows)

    n_math = int(args.n_total * args.math_frac)
    n_gsm = args.n_total - n_math
    rows = gsm_rows[:n_gsm] + math_rows[:n_math]
    rng.shuffle(rows)
    print("selected", len(rows))

    with open(args.out, "w") as fo:
        for prob, sol, ans, src in rows:
            user = MATH_PROMPT_TEMPLATE.format(prompt=prob.strip())
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, 10)
                shots = rng.sample(fs_pool, k)
                block = "\n\n".join(sample_to_fewshot(*s) for s in shots)
                user = block + "\n\n" + user
            resp = build_response(sol, ans)
            fo.write(json.dumps({"prompt": user, "response": resp,
                                 "source": src, "answer": ans,
                                 "problem": prob}) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
