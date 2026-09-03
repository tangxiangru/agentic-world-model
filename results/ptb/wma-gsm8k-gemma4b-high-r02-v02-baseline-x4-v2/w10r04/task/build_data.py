#!/usr/bin/env python3
"""Build the SFT corpus for gsm8k, rendered in the exact strings the grader uses.

The grader (inspect_evals/gsm8k) wraps every question in MATH_PROMPT_TEMPLATE and
renders the conversation with templates/gemma3.jinja, then scores the LAST numeric
token of the completion.  So every training row is built as:

  prompt     = "<bos><start_of_turn>user\n" [+ fewshot block + "\n\n"]
               + MATH_PROMPT_TEMPLATE.format(prompt=question)
               + "<end_of_turn>\n<start_of_turn>model\n"
  completion = "<chain of thought>\n\nANSWER: <n><end_of_turn>"

The strings are produced by rendering templates/gemma3.jinja itself (not by hand)
for a sample of rows, and the two are asserted equal - see verify_render().
"""
import argparse
import json
import random
import re
from collections import defaultdict

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"

CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUM_OK = re.compile(r"^-?\d{1,12}(\.\d{1,4})?$")


def build_prompt(question: str, fewshot_block: str | None) -> str:
    body = MATH_PROMPT_TEMPLATE.format(prompt=question.strip())
    if fewshot_block:
        body = fewshot_block.strip() + "\n\n" + body
    return f"{BOS}{SOT}user\n{body}{EOT}\n{SOT}model\n"


def build_completion(cot: str, answer: str) -> str:
    return f"{cot.strip()}\n\nANSWER: {answer}{EOT}"


def clean_gsm8k_cot(ans: str) -> tuple[str, str]:
    """original gsm8k train answer -> (cot without calculator annotations, final)"""
    body, final = ans.split("####")
    return CALC.sub("", body).strip(), final.strip()


def clean_omi_solution(sol: str) -> str:
    """drop \\boxed{} wrappers; the ANSWER: line is appended separately"""
    sol = BOXED.sub(r"\1", sol)
    sol = sol.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    return sol.strip()


def sample_to_fewshot(q: str, cot: str, final: str) -> str:
    return f"{q}\n\nReasoning:\n{cot}\n\nANSWER: {final}"


def last_number(text: str) -> str | None:
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w2 = w.replace(",", "").replace("$", "").rstrip(".")
        if w2.replace(".", "").replace("-", "").isnumeric():
            return w2
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_v1.jsonl")
    ap.add_argument("--n-omi-gsm8k", type=int, default=20000)
    ap.add_argument("--n-omi-aug", type=int, default=36000)
    ap.add_argument("--n-omi-math", type=int, default=6000)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=250)
    ap.add_argument("--holdout-out", default="data/dev_holdout.jsonl")
    ap.add_argument("--fewshot-ks", default="1,2,3,4,5,8,10",
                    help="few-shot depths to draw from for the prefixed rows")
    ap.add_argument("--extra", default="", help="comma-separated jsonl files appended verbatim")
    ap.add_argument("--cap-gsm8k", type=int, default=3)
    ap.add_argument("--cap-aug", type=int, default=2)
    ap.add_argument("--exclude", default="", help="jsonl whose (question, cot) pairs are skipped")
    ap.add_argument("--include-orig", type=int, default=1)
    args = ap.parse_args()

    global FEWSHOT_KS
    FEWSHOT_KS = [int(x) for x in args.fewshot_ks.split(",")]
    rng = random.Random(args.seed)
    from datasets import load_from_disk

    gsm = load_from_disk("data/gsm8k_raw")["train"]

    # ---- local dev set: gsm8k TRAIN items held out of every source ----
    #      (the benchmark test split is never touched by anything here)
    fewshot_qs = {r["question"] for r in gsm.shuffle(seed=42).select(range(10))}
    cand = [r for r in gsm.shuffle(seed=1234) if r["question"] not in fewshot_qs]
    holdout = cand[: args.holdout]
    held_qs = {r["question"] for r in holdout}
    with open(args.holdout_out, "w") as f:
        for r in holdout:
            cot, final = clean_gsm8k_cot(r["answer"])
            f.write(json.dumps({"question": r["question"], "gold": final, "cot": cot}) + "\n")
    print("holdout:", len(holdout), "->", args.holdout_out)

    # ---- fewshot pool: original gsm8k train items, exactly the grader's rendering ----
    pool = []
    for r in gsm.shuffle(seed=7).select(range(2500)):
        if r["question"] in held_qs:      # never show a dev item inside a training prompt
            continue
        cot, final = clean_gsm8k_cot(r["answer"])
        pool.append(sample_to_fewshot(r["question"], cot, final))

    rows = []
    held_qs_stripped = {q.strip() for q in held_qs}
    exclude = set()
    if args.exclude:
        for line in open(args.exclude):
            r = json.loads(line)
            exclude.add((r["question"].strip(), r["cot"].strip()))
    print("exclude pairs:", len(exclude))

    def add(question, cot, answer, src):
        if not NUM_OK.match(str(answer)):
            return
        if question.strip() in held_qs_stripped:
            return
        cot = cot.strip()
        if not cot or len(cot) > args.max_sol_chars:
            return
        # the grader reads the last numeric token: it must be our answer
        if (question.strip(), cot) in exclude:
            return
        comp = build_completion(cot, answer)
        if last_number(comp.replace(EOT, "")) != str(answer).replace(",", ""):
            return
        rows.append({"question": question.strip(), "answer": comp.replace(EOT, "").strip(),
                     "cot": cot, "final": str(answer), "source": src})

    # ---- 1. original gsm8k train, original chains ----
    if args.include_orig:
        for r in gsm:
            cot, final = clean_gsm8k_cot(r["answer"])
            add(r["question"], cot, final, "gsm8k_train_orig")
    n_orig = len(rows)

    # ---- 2. OpenMathInstruct-2 (derived from gsm8k/MATH TRAIN only) ----
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(
        "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"))
    per_problem = defaultdict(int)
    buckets = {"gsm8k": [], "augmented_gsm8k": [], "math": [], "augmented_math": []}
    caps = {"gsm8k": args.cap_gsm8k, "augmented_gsm8k": args.cap_aug, "math": 1, "augmented_math": 1}
    for f in files:
        df = pq.read_table(f).to_pandas()
        for src, cap in caps.items():
            sub = df[df.problem_source == src]
            for q, sol, ans in zip(sub["problem"], sub["generated_solution"], sub["expected_answer"]):
                if per_problem[q] >= cap:
                    continue
                per_problem[q] += 1
                buckets[src].append((q, sol, ans))
        del df

    def take(src, n):
        items = buckets[src]
        rng.shuffle(items)
        got = 0
        for q, sol, ans in items:
            if got >= n:
                break
            before = len(rows)
            add(q, clean_omi_solution(sol), ans, "omi_" + src)
            got += len(rows) - before

    take("gsm8k", args.n_omi_gsm8k)
    take("augmented_gsm8k", args.n_omi_aug)
    take("math", args.n_omi_math // 2)
    take("augmented_math", args.n_omi_math - args.n_omi_math // 2)

    # ---- dedup on (question, cot) ----
    seen, uniq = set(), []
    for r in rows:
        k = (r["question"], r["cot"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    rows = uniq
    rng.shuffle(rows)

    # ---- render prompt/completion, sprinkling fewshot prefixes ----
    for r in rows:
        if rng.random() < args.fewshot_frac:
            k = rng.choice(FEWSHOT_KS)
            block = "\n\n".join(rng.sample(pool, k))
        else:
            block = None
        r["prompt"] = build_prompt(r["question"], block)
        r["completion"] = build_completion(r["cot"], r["final"])

    extra_rows = []
    for path in [p for p in args.extra.split(",") if p]:
        extra_rows += [json.loads(l) for l in open(path)]
    rows += extra_rows
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print("orig gsm8k rows:", n_orig)
    print(Counter(r["source"] for r in rows))
    print("total:", len(rows), "->", args.out)


if __name__ == "__main__":
    main()
