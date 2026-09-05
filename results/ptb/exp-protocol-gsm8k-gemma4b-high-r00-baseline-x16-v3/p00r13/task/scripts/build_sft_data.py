#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> GSM8K.

Every target is shaped for the grader that evaluate.py actually runs:
inspect_evals/gsm8k with scorer=match(location="end", numeric=True), i.e. the
LAST numeric whitespace-token of the completion is the answer, and the gemma3
chat template stops the turn on <end_of_turn>.

Output JSONL rows: {"prompt": <rendered chat prefix>, "completion": <target + <end_of_turn>>}
The prompt string is the byte-exact render of templates/gemma3.jinja for the
same conversation the grader builds, so training and grading see one format.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent

# byte-for-byte from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{")


def unwrap_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    while True:
        m = BOXED_RE.search(text)
        if m is None:
            return text
        i = m.end()  # just after '{'
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth:  # unbalanced; give up
            return text
        text = text[: m.start()] + text[m.end() : i - 1] + text[i:]


def render_prompt(system: str | None, question: str) -> str:
    """Exactly what templates/gemma3.jinja produces for
    [system?, user] with add_generation_prompt=True."""
    user = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    prefix = (system + "\n\n") if system else ""
    return "<bos><start_of_turn>user\n" + prefix + user + "<end_of_turn>\n<start_of_turn>model\n"


def sample_to_fewshot(question: str, answer: str) -> str:
    """inspect_evals' own few-shot rendering, from (question, raw gsm8k answer)."""
    parts = answer.split("####")
    target = parts.pop().strip()
    reasoning = "####".join(parts).strip()
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--n-math", type=int, default=0,
                    help="extra augmented_math rows (numeric answers only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    from datasets import load_dataset

    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    fewshot_pool = [(q, a) for q, a in zip(gsm_train["question"], gsm_train["answer"])]

    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")

    src_col = ds["problem_source"]

    def collect(sources, want, seen_problem):
        rows, per_problem = [], {}
        idx = [i for i, s in enumerate(src_col) if s in sources]
        rng.shuffle(idx)
        sub = ds.select(idx)
        for r in sub:
            if len(rows) >= want:
                break
            ans = r["expected_answer"].strip().replace(",", "")
            if not NUMERIC_RE.match(ans):
                continue
            prob = r["problem"].strip()
            if prob in seen_problem:
                continue
            k = per_problem.get(prob, 0)
            if k >= args.max_per_problem:
                continue
            sol = unwrap_boxed(r["generated_solution"]).strip()
            if not sol or "ANSWER:" in sol:
                continue
            per_problem[prob] = k + 1
            rows.append((prob, sol, ans))
        return rows

    seen: set[str] = set()
    rows = collect({"gsm8k", "augmented_gsm8k"}, args.n, seen)
    seen |= {p for p, _, _ in rows}
    if args.n_math:
        rows += collect({"augmented_math", "math"}, args.n_math, seen)
    rng.shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_fs = 0
    with out.open("w") as f:
        for prob, sol, ans in rows:
            if rng.random() < args.fewshot_frac:
                shots = rng.sample(fewshot_pool, 10)
                system = "\n\n".join(sample_to_fewshot(q, a) for q, a in shots)
                n_fs += 1
            else:
                system = None
            completion = sol + "\n\nANSWER: " + ans + "<end_of_turn>"
            f.write(json.dumps({
                "prompt": render_prompt(system, prob),
                "completion": completion,
            }) + "\n")
    print(f"wrote {len(rows)} rows to {out} ({n_fs} with a 10-shot prefix)")


if __name__ == "__main__":
    main()
