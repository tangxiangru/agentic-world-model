#!/usr/bin/env python3
"""Build the SFT corpus for gemma-3-4b-pt -> inspect_evals/gsm8k.

Everything here is shaped so that the string the trainer sees is byte-identical
to the string the grader renders with templates/gemma3.jinja:

  <bos><start_of_turn>user
  [<k few-shot examples>\n\n]Solve the following math problem step by step. ...
  <end_of_turn>
  <start_of_turn>model
  <reasoning>
  ANSWER: <n><end_of_turn>

Sources are all derived from GSM8K *train* / MATH *train* (never the test split).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

import pyarrow.parquet as pq

# --- the grader's own strings, copied verbatim from inspect_evals/gsm8k -------
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI2 = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
GSM8K_TRAIN = "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"

BOXED = re.compile(r"\\boxed\{")
CALC = re.compile(r"<<[^>]*>>")


def strip_boxed(text: str) -> str:
    """Replace every \\boxed{...} with its contents (brace-balanced)."""
    out = []
    i = 0
    while True:
        m = BOXED.search(text, i)
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
        out.append(text[m.end():j - 1])
        i = j
    return "".join(out)


def numeric(ans: str) -> str | None:
    """Return the answer normalised to a plain number string, else None."""
    a = ans.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if re.fullmatch(r"-?\d+", a):
        return a
    if re.fullmatch(r"-?\d*\.\d+", a):
        return a.rstrip("0").rstrip(".") if "." in a else a
    return None


def render_prompt(question: str, shots: list[tuple[str, str, str]]) -> str:
    """shots: list of (question, reasoning, answer) rendered into the system slot."""
    user = MATH_PROMPT_TEMPLATE.format(prompt=question)
    if shots:
        sysmsg = "\n\n".join(
            f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}" for q, r, a in shots
        )
        user = sysmsg + "\n\n" + user
    return "<bos><start_of_turn>user\n" + user.strip() + "<end_of_turn>\n<start_of_turn>model\n"


def render_completion(reasoning: str, answer: str) -> str:
    return reasoning.strip() + f"\n\nANSWER: {answer}<end_of_turn>\n"


def load_gsm8k_train() -> list[tuple[str, str, str]]:
    path = glob.glob(GSM8K_TRAIN)[0]
    rows = pq.read_table(path).to_pylist()
    out = []
    for r in rows:
        body, _, ans = r["answer"].partition("####")
        out.append((r["question"].strip(), CALC.sub("", body).strip(), ans.strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm", type=int, default=115000)
    ap.add_argument("--n-math", type=int, default=35000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm_train = load_gsm8k_train()

    files = sorted(glob.glob(OMI2))
    assert files, "OpenMathInstruct-2 not downloaded"

    buckets: dict[str, list[dict]] = defaultdict(list)
    per_problem: dict[str, int] = defaultdict(int)
    want_gsm, want_math = args.n_gsm, args.n_math
    # oversample from disk, then subsample -- keeps the mixture from being
    # dominated by whichever shard happened to come first.
    for f in files:
        t = pq.read_table(f, columns=["problem", "generated_solution",
                                      "expected_answer", "problem_source"])
        for r in t.to_pylist():
            src = r["problem_source"]
            fam = "gsm" if "gsm8k" in src else "math"
            if fam == "gsm" and len(buckets["gsm"]) >= want_gsm * 1.6:
                if len(buckets["math"]) >= want_math * 1.6:
                    break
                continue
            if fam == "math" and len(buckets["math"]) >= want_math * 1.6:
                continue
            ans = numeric(r["expected_answer"])
            if ans is None:
                continue
            key = r["problem"]
            if per_problem[key] >= args.max_per_problem:
                continue
            sol = strip_boxed(r["generated_solution"]).strip()
            if not sol or len(sol) > 4000:
                continue
            # the reasoning must not carry a second answer marker
            if "ANSWER:" in sol or "####" in sol:
                continue
            per_problem[key] += 1
            buckets[fam].append({"q": key.strip(), "r": sol, "a": ans})
        if len(buckets["gsm"]) >= want_gsm * 1.6 and len(buckets["math"]) >= want_math * 1.6:
            break
        print(f"{f.split('/')[-1]}: gsm={len(buckets['gsm'])} math={len(buckets['math'])}",
              flush=True)

    rng.shuffle(buckets["gsm"])
    rng.shuffle(buckets["math"])
    rows = buckets["gsm"][:want_gsm] + buckets["math"][:want_math]
    rng.shuffle(rows)
    print(f"kept gsm={min(len(buckets['gsm']), want_gsm)} "
          f"math={min(len(buckets['math']), want_math)} total={len(rows)}")

    n_fs = 0
    with open(args.out, "w") as fh:
        for row in rows:
            shots: list[tuple[str, str, str]] = []
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, 4)
                shots = rng.sample(gsm_train, k)
                n_fs += 1
            fh.write(json.dumps({
                "prompt": render_prompt(row["q"], shots),
                "completion": render_completion(row["r"], row["a"]),
                "answer": row["a"],
            }) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows, {n_fs} with a few-shot prefix")


if __name__ == "__main__":
    main()
