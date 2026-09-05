"""Build a second-stage SFT mix in the grader's rendered format.

Three streams, all GSM8K-train-derived or independent public word problems;
no benchmark test item is involved:

  omi_new   nvidia/OpenMathInstruct-2 (full split), gsm8k/augmented_gsm8k rows
            whose problem was NOT used in the first-stage file
  omi_alt   additional distinct solutions to problems that WERE used, capped at
            --alt-per-problem (the dataset ships many verified solutions each)
  orca      microsoft/orca-math-word-problems-200k, an independent 200k set of
            grade-school word problems; the gold answer is the last number of
            the reference solution, which is exactly what the grader reads
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
LASTNUM_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    if "." in a:
        f = float(a)
        if f.is_integer():
            a = str(int(f))
    return a


def omi_clean(sol: str, answer: str) -> str | None:
    if "\\boxed" not in sol:
        return None
    body, _ = BOXED_RE.subn(r"\1", sol)
    if "\\boxed" in body:
        return None
    body = body.strip()
    return f"{body}\n\nANSWER: {answer}" if body else None


def orca_clean(sol: str) -> tuple[str, str] | None:
    """Gold = last number of the reference solution; require it to sit in the
    final sentence so we do not label an intermediate value as the answer."""
    body = sol.strip()
    if len(body) < 40 or len(body) > 2600:
        return None
    if "\\boxed" in body or "ANSWER:" in body:
        return None
    nums = LASTNUM_RE.findall(body)
    if not nums:
        return None
    a = norm_answer(nums[-1])
    if a is None:
        return None
    tail = body[max(0, len(body) - 220):]
    if nums[-1] not in tail:
        return None
    return body, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--used", default="data/sft_v1_questions.jsonl")
    ap.add_argument("--n-omi-new", type=int, default=20000)
    ap.add_argument("--n-omi-alt", type=int, default=40000)
    ap.add_argument("--alt-per-problem", type=int, default=2)
    ap.add_argument("--n-orca", type=int, default=45000)
    ap.add_argument("--n-math", type=int, default=8000)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--max-sol-chars", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open("data/dev_train300.jsonl") as f:
        dev_q = {json.loads(l)["question"].strip() for l in f}
    used_q = set()
    with open(args.used) as f:
        for l in f:
            used_q.add(json.loads(l)["question"].strip())
    fewshot_q = {r["question"].strip() for r in
                 load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(10))}
    banned = dev_q | fewshot_q
    print(f"[mix] {len(used_q)} first-stage questions, {len(banned)} banned", flush=True)

    rows = []
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train")

    def omi_stream(sources, cap, allow_used, per_problem):
        sub = ds.filter(lambda r: r["problem_source"] in sources, num_proc=16)
        idx = list(range(len(sub)))
        rng.shuffle(idx)
        out, cnt = [], {}
        for i in idx:
            r = sub[i]
            q = r["problem"].strip()
            if q in banned:
                continue
            if (q in used_q) != allow_used:
                continue
            if cnt.get(q, 0) >= per_problem:
                continue
            a = norm_answer(r["expected_answer"])
            if a is None:
                continue
            sol = r["generated_solution"]
            if len(sol) > args.max_sol_chars:
                continue
            body = omi_clean(sol, a)
            if body is None:
                continue
            cnt[q] = cnt.get(q, 0) + 1
            out.append((q, body))
            if len(out) >= cap:
                break
        return out

    g = ("gsm8k", "augmented_gsm8k")
    a1 = omi_stream(g, args.n_omi_new, allow_used=False, per_problem=1)
    print(f"[mix] omi_new  {len(a1)}", flush=True)
    a2 = omi_stream(g, args.n_omi_alt, allow_used=True, per_problem=args.alt_per_problem)
    print(f"[mix] omi_alt  {len(a2)}", flush=True)
    a4 = omi_stream(("math", "augmented_math"), args.n_math, allow_used=False, per_problem=1)
    print(f"[mix] omi_math {len(a4)}", flush=True)
    rows += a1 + a2 + a4

    if args.n_orca > 0:
        orca = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
        idx = list(range(len(orca)))
        rng.shuffle(idx)
        seen, got = set(), 0
        for i in idx:
            r = orca[i]
            q = r["question"].strip()
            if q in banned or q in seen or q in used_q:
                continue
            c = orca_clean(r["answer"])
            if c is None:
                continue
            body, a = c
            seen.add(q)
            rows.append((q, f"{body}\n\nANSWER: {a}"))
            got += 1
            if got >= args.n_orca:
                break
        print(f"[mix] orca     {got}", flush=True)

    sysmsg = gf.build_fewshot_system()
    rng.shuffle(rows)
    n_few = int(len(rows) * args.fewshot_frac)
    with open(args.out, "w") as f:
        for i, (q, body) in enumerate(rows):
            system = sysmsg if i < n_few else None
            f.write(json.dumps({"prompt": gf.render_prompt(q, system),
                                "completion": gf.render_target(body),
                                "fewshot": bool(system), "question": q}) + "\n")
    print(f"[mix] wrote {len(rows)} rows to {args.out} ({n_few} with the 10-shot prefix)")


if __name__ == "__main__":
    main()
