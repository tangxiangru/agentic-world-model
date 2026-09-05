"""Build the SFT pool for GSM8K.

Every row is rendered with templates/gemma3.jinja (the grader's own template) so
training and grading see byte-identical strings.  Targets end with exactly one
"ANSWER: <n>" line and nothing after it, because the grader
(inspect_ai match(location="end", numeric=True)) reads the LAST number in the
completion.
"""
import argparse, hashlib, json, random, re, sys
from datasets import load_dataset

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")
CALC_RE = re.compile(r"<<[^>]*>>")


def norm_num(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return int(f) if f == int(f) else round(f, 6)


def last_number(text):
    ms = NUM_RE.findall(text)
    if not ms:
        return None
    return norm_num(ms[-1])


def clean_boxed(sol):
    """Replace \\boxed{X} with X and drop trailing 'The answer is' chatter."""
    out, i = [], 0
    while True:
        j = sol.find("\\boxed{", i)
        if j < 0:
            out.append(sol[i:])
            break
        out.append(sol[i:j])
        k, depth = j + 7, 1
        while k < len(sol) and depth:
            if sol[k] == "{":
                depth += 1
            elif sol[k] == "}":
                depth -= 1
            k += 1
        out.append(sol[j + 7 : k - 1])
        i = k
    return "".join(out)


def clean_solution(sol):
    sol = CALC_RE.sub("", sol)
    sol = clean_boxed(sol)
    sol = sol.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    # strip any pre-existing answer marker lines so ours is the only one
    lines = []
    for ln in sol.split("\n"):
        s = ln.strip()
        low = s.lower()
        if low.startswith("the answer is") or low.startswith("answer:") or s.startswith("####"):
            continue
        lines.append(ln.rstrip())
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def make_row(question, solution, answer):
    """Return (prompt_user_text, target_text) or None if the row is unusable."""
    ans = norm_num(str(answer))
    if ans is None:
        return None
    body = clean_solution(solution)
    if len(body) < 20 or len(body) > 4000:
        return None
    if "ANSWER:" in body or "\\boxed" in body:
        return None
    target = f"{body}\n\nANSWER: {ans}"
    # the grader reads the last number: it must be the gold answer
    if last_number(target) != ans:
        return None
    if target.count("ANSWER:") != 1:
        return None
    return PROMPT_TEMPLATE.format(prompt=question.strip()), target


def fewshot_prefix(pool, k, rng):
    picks = rng.sample(pool, k)
    return "\n\n".join(picks) + "\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_pool.jsonl")
    ap.add_argument("--omi-gsm8k", type=int, default=153000)
    ap.add_argument("--omi-math", type=int, default=25000)
    ap.add_argument("--orca", type=int, default=70000)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- few-shot exemplars, formatted exactly like the grader's own ----------
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    fs_pool = []
    for r in gsm:
        q = r["question"].strip()
        a = r["answer"]
        reasoning, _, tgt = a.rpartition("####")
        reasoning = CALC_RE.sub("", reasoning).strip()
        fs_pool.append(f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {tgt.strip()}")
    print(f"fewshot pool: {len(fs_pool)}", flush=True)

    rows, seen = [], set()

    def add(question, solution, answer, src):
        r = make_row(question, solution, answer)
        if r is None:
            return False
        key = hashlib.md5((question.strip().lower()).encode()).hexdigest()
        if key in seen:
            return False
        seen.add(key)
        rows.append({"prompt": r[0], "target": r[1], "source": src,
                     "question": question.strip()})
        return True

    # ---- OpenMathInstruct-2 --------------------------------------------------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    n_g = n_m = 0
    idx = list(range(len(omi)))
    rng.shuffle(idx)
    for i in idx:
        r = omi[i]
        ps = r["problem_source"]
        if ps in ("gsm8k", "augmented_gsm8k"):
            if n_g < args.omi_gsm8k and add(r["problem"], r["generated_solution"],
                                            r["expected_answer"], "omi2_gsm8k"):
                n_g += 1
        elif ps in ("math", "augmented_math"):
            if n_m < args.omi_math and add(r["problem"], r["generated_solution"],
                                           r["expected_answer"], "omi2_math"):
                n_m += 1
        if n_g >= args.omi_gsm8k and n_m >= args.omi_math:
            break
    print(f"omi2 gsm8k={n_g} math={n_m}", flush=True)

    # ---- Orca-Math -----------------------------------------------------------
    orca = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    oidx = list(range(len(orca)))
    rng.shuffle(oidx)
    n_o = 0
    for i in oidx:
        if n_o >= args.orca:
            break
        r = orca[i]
        ans = last_number(r["answer"])
        if ans is None:
            continue
        if add(r["question"], r["answer"], ans, "orca_math"):
            n_o += 1
    print(f"orca={n_o}", flush=True)

    # ---- few-shot dressing on a slice ---------------------------------------
    rng.shuffle(rows)
    n_fs = int(len(rows) * args.fewshot_frac)
    for r in rows[:n_fs]:
        k = rng.choice([1, 2, 3, 4])
        r["prompt"] = fewshot_prefix(fs_pool, k, rng) + r["prompt"]
        r["fewshot"] = k
    for r in rows[n_fs:]:
        r["fewshot"] = 0

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
