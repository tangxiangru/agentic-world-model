#!/usr/bin/env python3
"""Build SFT data for GSM8K in the grader's exact prompt/target format.

Sources (all GSM8K *train*-derived or original train solutions; never the test split):
  - openai/gsm8k, split=train            (7473 reference solutions)
  - nvidia/OpenMathInstruct-2, train_1M  (problem_source in {gsm8k, augmented_gsm8k})

Output: jsonl rows {"prompt": <user turn content>, "completion": <model turn text>,
                    "answer": <int as string>, "source": <str>, "text": prompt+completion}
The "text" field is what ../contamination_check.py reads.
"""
import argparse, json, random, re, collections, os

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

CALC_RE = re.compile(r"<<[^>]*>>")
BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{X} with X (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        j = m.end()
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        out.append(text[m.end(): j - 1])
        i = j
    return "".join(out)


def norm_int(s: str):
    """Return the canonical integer string, or None if not a plain integer."""
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    if s.endswith(".0"):
        s = s[:-2]
    if re.fullmatch(r"-?\d+", s):
        v = int(s)
        if abs(v) < 10 ** 12:
            return str(v)
    return None


def make_row(question: str, solution: str, answer: str, source: str):
    prompt = MATH_PROMPT_TEMPLATE.format(prompt=question.strip())
    completion = f"{solution.strip()}\n\nANSWER: {answer}"
    return {
        "prompt": prompt,
        "completion": completion,
        "answer": answer,
        "source": source,
        "text": question.strip() + "\n" + solution.strip() + "\nANSWER: " + answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=4)
    ap.add_argument("--max-aug", type=int, default=10 ** 9,
                    help="cap on augmented_gsm8k rows kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gsm8k-repeat", type=int, default=1,
                    help="how many times to include the original gsm8k train solutions")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from datasets import load_dataset

    rows = []

    # ---- 1. original GSM8K train reference solutions -----------------------
    g = load_dataset("openai/gsm8k", "main", split="train")
    n_g = 0
    for r in g:
        body, _, ans = r["answer"].rpartition("####")
        a = norm_int(ans)
        if a is None:
            continue
        sol = CALC_RE.sub("", body).strip()
        if not sol:
            continue
        for _ in range(args.gsm8k_repeat):
            rows.append(make_row(r["question"], sol, a, "gsm8k_train_ref"))
        n_g += 1
    print(f"gsm8k train reference solutions: {n_g} problems -> {len(rows)} rows")

    # ---- 2. OpenMathInstruct-2 gsm8k-derived -------------------------------
    d = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    keep_sources = {"gsm8k", "augmented_gsm8k"}
    per_problem = collections.Counter()
    seen = set()
    n_bad_ans, n_dup, n_cap = 0, 0, 0
    aug_rows = []
    for r in d:
        src = r["problem_source"]
        if src not in keep_sources:
            continue
        a = norm_int(r["expected_answer"])
        if a is None:
            n_bad_ans += 1
            continue
        sol = strip_boxed(r["generated_solution"]).strip()
        if not sol or len(sol) > 6000:
            continue
        key = (r["problem"], sol)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        if per_problem[r["problem"]] >= args.max_per_problem:
            n_cap += 1
            continue
        per_problem[r["problem"]] += 1
        aug_rows.append(make_row(r["problem"], sol, a, "omi2_" + src))
    print(f"omi2 gsm8k-derived kept={len(aug_rows)} bad_answer={n_bad_ans} "
          f"dup={n_dup} over_cap={n_cap} unique_problems={len(per_problem)}")

    rng.shuffle(aug_rows)
    if len(aug_rows) > args.max_aug:
        aug_rows = aug_rows[: args.max_aug]
    rows.extend(aug_rows)

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(collections.Counter(r["source"] for r in rows))


if __name__ == "__main__":
    main()
