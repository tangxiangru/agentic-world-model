#!/usr/bin/env python3
"""Build SFT data for GSM8K from OpenMathInstruct-2 + GSM8K train split."""
import json, random, re, os, argparse
from collections import defaultdict
from datasets import load_from_disk, load_dataset

random.seed(0)

BOXED = re.compile(r"\\boxed\{")


def strip_boxed(sol: str) -> str:
    """Replace \\boxed{X} with X (handles nesting)."""
    out = []
    i = 0
    while True:
        m = BOXED.search(sol, i)
        if not m:
            out.append(sol[i:])
            break
        out.append(sol[i:m.start()])
        j = m.end()
        depth = 1
        while j < len(sol) and depth:
            if sol[j] == '{':
                depth += 1
            elif sol[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(sol[m.end():j])
        i = j + 1
    return "".join(out)


def clean_gsm8k_solution(ans: str) -> tuple[str, str]:
    """Original GSM8K answer -> (reasoning without calc annotations, final)."""
    body, final = ans.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return body, final.strip()


def is_numeric_answer(a: str) -> bool:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    if a.startswith("-"):
        a = a[1:]
    return a.replace(".", "", 1).isdigit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/sft_data.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--n-math", type=int, default=25000)
    args = ap.parse_args()

    omi = load_from_disk("work/data/omi2_1M").data.table
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")

    records = []

    # ---- 1. Original GSM8K train solutions (target style / distribution) ----
    for r in gsm_train:
        body, final = clean_gsm8k_solution(r["answer"])
        records.append({
            "question": r["question"],
            "solution": body,
            "answer": final,
            "src": "gsm8k_orig",
        })

    # ---- 2. OpenMathInstruct-2, gsm8k-derived ----
    by_problem = defaultdict(list)
    math_pool = []
    src = omi.column("problem_source").to_pylist()
    probs = omi.column("problem").to_pylist()
    sols = omi.column("generated_solution").to_pylist()
    answers = omi.column("expected_answer").to_pylist()
    print("columns loaded", flush=True)
    for i in range(len(src)):
        s = src[i]
        if s in ("gsm8k", "augmented_gsm8k"):
            by_problem[probs[i]].append(i)
        elif s in ("math", "augmented_math"):
            math_pool.append(i)

    print("unique gsm8k-derived problems:", len(by_problem))
    for p, idxs in by_problem.items():
        random.shuffle(idxs)
        for i in idxs[:args.max_per_problem]:
            a = answers[i]
            if not is_numeric_answer(a):
                continue
            records.append({
                "question": p,
                "solution": strip_boxed(sols[i]).strip(),
                "answer": a.strip(),
                "src": src[i],
            })

    # ---- 3. Some MATH-derived for broader reasoning (numeric answers only) ----
    random.shuffle(math_pool)
    n = 0
    seen_math = set()
    for i in math_pool:
        if n >= args.n_math:
            break
        a = answers[i]
        if not is_numeric_answer(a):
            continue
        if probs[i] in seen_math:
            continue
        seen_math.add(probs[i])
        records.append({
            "question": probs[i],
            "solution": strip_boxed(sols[i]).strip(),
            "answer": a.strip(),
            "src": src[i],
        })
        n += 1

    # filter overly long / degenerate
    out = []
    for r in records:
        if not r["solution"] or len(r["solution"]) > 4000:
            continue
        if "\\boxed" in r["solution"]:
            continue
        out.append(r)

    random.shuffle(out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(Counter(r["src"] for r in out))
    print("total", len(out), "->", args.out)


if __name__ == "__main__":
    main()
