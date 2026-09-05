#!/usr/bin/env python3
"""Build the SFT dataset for GSM8K from OpenMathInstruct-2 (gsm8k + augmented_gsm8k).

Output: data/sft.jsonl with fields {"prompt_text", "completion_text", "question", "answer"}
The prompt text is the *exact* string the eval will feed the model (gemma3.jinja render).
"""
import json
import random
import re
import argparse
from collections import defaultdict

from datasets import load_from_disk

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def render_prompt(question: str, system: str | None = None) -> str:
    """Reproduce templates/gemma3.jinja for a single user turn (+ optional system)."""
    user = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    prefix = (system.strip() + "\n\n") if system else ""
    return "<bos><start_of_turn>user\n" + prefix + user + "<end_of_turn>\n<start_of_turn>model\n"


NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    if not NUM_RE.match(a):
        return None
    if "." in a:
        f = float(a)
        if f == int(f):
            return str(int(f))
        return a.rstrip("0").rstrip(".")
    return str(int(a))


BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(s: str) -> str:
    """Replace \\boxed{X} with X (brace-balanced)."""
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


LATEX_CMDS = [
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\!|\\,|\\;|\\quad|\\qquad"), " "),
    (re.compile(r"\\times"), "x"),
    (re.compile(r"\\cdot"), "*"),
    (re.compile(r"\\div"), "/"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\\$"), "$"),
    (re.compile(r"\\text\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}"), r"\1/\2"),
    (re.compile(r"\\dfrac\{([^{}]+)\}\{([^{}]+)\}"), r"\1/\2"),
    (re.compile(r"\\\[|\\\]"), ""),
    (re.compile(r"\\\(|\\\)"), ""),
]


def clean_solution(sol: str) -> str:
    s = strip_boxed(sol)
    for pat, rep in LATEX_CMDS:
        s = pat.sub(rep, s)
    s = s.replace("$", "")
    # collapse whitespace inside lines, drop blank-line runs
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    out, blank = [], False
    for ln in lines:
        if not ln:
            blank = True
            continue
        if blank and out:
            out.append("")
        blank = False
        out.append(ln)
    return "\n".join(out).strip()


BAD_TOKENS = ("\\begin", "\\end", "\\sqrt", "\\pi", "\\sum", "\\int", "\\ge", "\\le",
              "\\approx", "\\neq", "\\alpha", "\\beta", "\\theta", "^{", "_{", "\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    omi = load_from_disk("data/omi2_gsm")
    gsm_train = load_from_disk("data/gsm8k_main")["train"]

    # fewshot pool built from the GSM8K *train* split, exactly like inspect's sample_to_fewshot
    fewshot_pool = []
    for r in gsm_train:
        ans = r["answer"].split("####")
        target = ans.pop().strip()
        reasoning = "####".join(ans).strip()
        fewshot_pool.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")

    by_problem = defaultdict(list)
    n_bad_ans = n_bad_sol = 0
    for r in omi:
        ans = norm_answer(r["expected_answer"])
        if ans is None:
            n_bad_ans += 1
            continue
        sol = clean_solution(r["generated_solution"])
        if any(t in sol for t in BAD_TOKENS) or len(sol) < 30 or len(sol) > 2500:
            n_bad_sol += 1
            continue
        # the cleaned solution must actually contain the final answer somewhere
        if ans not in sol.replace(",", ""):
            n_bad_sol += 1
            continue
        by_problem[r["problem"].strip()].append((sol, ans))

    print(f"problems={len(by_problem)} dropped_ans={n_bad_ans} dropped_sol={n_bad_sol}")

    records = []
    for prob, sols in by_problem.items():
        rng.shuffle(sols)
        # keep only solutions agreeing with the majority answer
        from collections import Counter
        maj = Counter(a for _, a in sols).most_common(1)[0][0]
        sols = [s for s in sols if s[1] == maj]
        # prefer shorter, well-structured solutions
        sols.sort(key=lambda s: len(s[0]))
        for sol, ans in sols[: args.max_per_problem]:
            completion = sol.rstrip()
            # remove a trailing restatement line if it already ends with the answer sentence
            completion = completion + f"\n\nANSWER: {ans}"
            use_fs = rng.random() < args.fewshot_frac
            system = None
            if use_fs:
                k = rng.choice([1, 2, 3])
                system = "\n\n".join(rng.sample(fewshot_pool, k))
            records.append({
                "prompt_text": render_prompt(prob, system),
                "completion_text": completion + "<end_of_turn>",
                "question": prob,
                "answer": ans,
            })

    rng.shuffle(records)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
