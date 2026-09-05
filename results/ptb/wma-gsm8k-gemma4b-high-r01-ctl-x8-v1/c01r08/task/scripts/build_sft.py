"""Build the SFT file in the grader's exact rendered format.

Source: nvidia/OpenMathInstruct-2 (train_1M), rows whose problem_source is
gsm8k / augmented_gsm8k (both are GSM8K *train*-derived; no test item is
involved), plus an optional slice of math/augmented_math for robustness.

Every row is emitted as {"prompt": ..., "completion": ...} where the two
strings concatenate to exactly what templates/gemma3.jinja renders for the
conversation the grader will send, and the completion ends with <end_of_turn>.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grader_format as gf  # noqa: E402

from datasets import load_dataset  # noqa: E402

NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def clean_solution(sol: str, answer: str) -> str | None:
    """Strip \\boxed{} markup and pin one final 'ANSWER: n' line."""
    if "\\boxed" not in sol:
        return None
    # \boxed{x} -> x  (only single-level braces; rows with nested braces are dropped)
    body, n = BOXED_RE.subn(r"\1", sol)
    if "\\boxed" in body:
        return None
    body = body.strip()
    if not body:
        return None
    return f"{body}\n\nANSWER: {answer}"


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "")
    if not NUM_RE.match(a):
        return None
    # keep integers integral: "14.0" -> "14"
    if "." in a:
        f = float(a)
        if f.is_integer():
            a = str(int(f))
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm", type=int, default=60000)
    ap.add_argument("--n-math", type=int, default=6000)
    ap.add_argument("--fewshot-frac", type=float, default=0.08,
                    help="fraction of rows rendered with the grader's exact 10-shot system prefix")
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train_1M")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="jsonl files whose 'question' values must not be reused")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(os.path.join(os.path.dirname(args.out) or ".", "dev_train300.jsonl")) as f:
        dev_q = {json.loads(l)["question"].strip() for l in f}
    for xp in args.exclude:
        with open(xp) as f:
            for l in f:
                dev_q.add(json.loads(l)["question"].strip())
    print(f"holding out {len(dev_q)} questions (probe + excluded)", flush=True)

    sysmsg = gf.build_fewshot_system()
    fewshot_q = set()
    from datasets import load_dataset as _ld
    for r in _ld("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(10)):
        fewshot_q.add(r["question"].strip())

    ds = load_dataset("nvidia/OpenMathInstruct-2", split=args.split)

    def pick(sources, cap):
        sub = ds.filter(lambda r: r["problem_source"] in sources, num_proc=16)
        idx = list(range(len(sub)))
        rng.shuffle(idx)
        out, seen = [], set()
        for i in idx:
            r = sub[i]
            q = r["problem"].strip()
            if q in dev_q or q in fewshot_q:
                continue
            if q in seen:
                continue
            a = norm_answer(r["expected_answer"])
            if a is None:
                continue
            sol = r["generated_solution"]
            if len(sol) > args.max_sol_chars:
                continue
            body = clean_solution(sol, a)
            if body is None:
                continue
            seen.add(q)
            out.append((q, body))
            if len(out) >= cap:
                break
        return out

    rows = pick(("gsm8k", "augmented_gsm8k"), args.n_gsm)
    print(f"gsm8k-flavoured rows: {len(rows)}", flush=True)
    if args.n_math > 0:
        m = pick(("math", "augmented_math"), args.n_math)
        print(f"math rows: {len(m)}", flush=True)
        rows += m

    rng.shuffle(rows)
    n_few = int(len(rows) * args.fewshot_frac)
    with open(args.out, "w") as f:
        for i, (q, body) in enumerate(rows):
            system = sysmsg if i < n_few else None
            f.write(json.dumps({
                "prompt": gf.render_prompt(q, system),
                "completion": gf.render_target(body),
                "fewshot": bool(system),
                "question": q,
            }) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} ({n_few} with the 10-shot prefix)")


if __name__ == "__main__":
    main()
