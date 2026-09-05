"""Build the SFT corpus for exp-02.

Source: nvidia/OpenMathInstruct-2 train_1M, rows whose problem_source is
gsm8k / augmented_gsm8k (i.e. derived from the GSM8K TRAIN split only).

Every row is rendered with the grader's own gemma3 chat template, and every
target ends with a bare "ANSWER: <n>" line followed by <end_of_turn>.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys

sys.path.insert(0, "/home/ben/task/scripts")
from common import (ANSWER_MARKER, grader_fewshot_system, get_tokenizer,  # noqa: E402
                    render_completion, render_prompt)

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{")


def strip_boxed(sol: str) -> str | None:
    """Replace the single \\boxed{...} with its contents (brace-balanced)."""
    m = BOXED_RE.search(sol)
    if m is None or len(BOXED_RE.findall(sol)) != 1:
        return None
    i = m.end()  # first char after '{'
    depth = 1
    j = i
    while j < len(sol) and depth:
        if sol[j] == "{":
            depth += 1
        elif sol[j] == "}":
            depth -= 1
        j += 1
    if depth:
        return None
    return sol[: m.start()] + sol[i: j - 1] + sol[j:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ben/task/data/sft_exp02.jsonl")
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--max-per-problem-orig", type=int, default=2)
    ap.add_argument("--max-sol-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_from_disk

    rng = random.Random(args.seed)
    tok = get_tokenizer()
    sysmsg = grader_fewshot_system()

    ds = load_from_disk("/home/ben/task/data/omi2_gsm")
    if args.limit:
        ds = ds.select(range(args.limit))

    per_problem: dict[str, int] = {}
    seen: set[str] = set()
    kept, drop = [], {"answer": 0, "boxed": 0, "marker": 0, "dup": 0, "cap": 0, "long": 0}

    for r in ds:
        prob, sol, ans, src = (r["problem"], r["generated_solution"],
                               r["expected_answer"], r["problem_source"])
        ans = ans.replace(",", "").strip()
        if not NUM_RE.match(ans):
            drop["answer"] += 1
            continue
        body = strip_boxed(sol)
        if body is None:
            drop["boxed"] += 1
            continue
        if ANSWER_MARKER in body or "####" in body:
            drop["marker"] += 1
            continue
        key = prob + "\x00" + body
        if key in seen:
            drop["dup"] += 1
            continue
        cap = args.max_per_problem_orig if src == "gsm8k" else args.max_per_problem
        if per_problem.get(prob, 0) >= cap:
            drop["cap"] += 1
            continue
        seen.add(key)
        per_problem[prob] = per_problem.get(prob, 0) + 1
        kept.append({"problem": prob, "body": body.strip(), "answer": ans, "src": src})

    print(f"kept {len(kept)}  dropped {drop}", flush=True)

    rng.shuffle(kept)
    n_few = int(len(kept) * args.fewshot_frac)

    n_long = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(kept):
            completion = render_completion(r["body"], r["answer"])
            n_ct = len(tok(completion, add_special_tokens=False)["input_ids"])
            if n_ct > args.max_sol_tokens:
                n_long += 1
                continue
            use_few = i < n_few
            prompt = render_prompt(tok, r["problem"], system=sysmsg if use_few else None)
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                "fewshot": use_few,
                "src": r["src"],
                # plain text of the training target, for the preflight checks
                "target": completion,
            }) + "\n")
    print(f"dropped {n_long} over-long completions; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
