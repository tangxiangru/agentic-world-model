#!/usr/bin/env python3
"""Build the SFT corpus for GSM8K post-training of gemma-3-4b-pt.

Targets are rendered with the *grader's* chat template (templates/gemma3.jinja),
so the string the trainer optimises is byte-identical to the string vLLM will be
prompted with at eval time.  Every target ends with a single "ANSWER: <n>" line
followed by <end_of_turn> (token 106), which is in the base model's
generation_config eos list, so vLLM stops there.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def fewshot_block(question: str, reasoning: str, answer: str) -> str:
    """Exactly inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {answer}"


def unbox(sol: str) -> str | None:
    """Replace the single \\boxed{...} with its contents.

    OpenMathInstruct-2 marks the final answer with \\boxed{}; the grader here
    reads the last number instead, so the marker has to go. Rows with zero or
    more than one \\boxed are dropped rather than guessed at.
    """
    if sol.count("\\boxed") != 1:
        return None
    i = sol.index("\\boxed")
    j = i + len("\\boxed")
    while j < len(sol) and sol[j] == " ":
        j += 1
    if j >= len(sol) or sol[j] != "{":
        return None
    depth = 0
    k = j
    while k < len(sol):
        if sol[k] == "{":
            depth += 1
        elif sol[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    if k >= len(sol):
        return None
    inner = sol[j + 1 : k]
    out = sol[:i] + inner + sol[k + 1 :]
    # tidy up the leftovers of "$\boxed{7}$" style wrappers
    out = out.replace("$$", "")
    return out.rstrip()


def clean_ok(text: str) -> bool:
    if "\\boxed" in text:
        return False
    if "ANSWER:" in text:
        return False
    if "####" in text:
        return False
    return True


def norm_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").replace("%", "")
    a = a.rstrip(".")
    if not a:
        return None
    try:
        v = float(a)
    except ValueError:
        return None
    if v == int(v):
        return str(int(v))
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi2", default="/home/ben/task/data/omi2_1M.parquet")
    ap.add_argument("--gsm-rest", default="/home/ben/task/data/gsm8k_train_rest.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi-gsm", type=int, default=70000)
    ap.add_argument("--n-omi-math", type=int, default=10000)
    ap.add_argument("--gsm-orig-repeat", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.20)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="previously built jsonl files whose (question, target) pairs must not reappear")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    excluded: set[str] = set()
    for p in args.exclude:
        for line in open(p):
            d = json.loads(line)
            excluded.add(d["user"] + "\x00" + d["target"])
    if excluded:
        print("excluding", len(excluded), "already-used (prompt, target) pairs")

    # ---- the pool of few-shot exemplars: GSM8K train, minus the held-out probe
    gsm_rest = [json.loads(l) for l in open(args.gsm_rest)]
    exemplars = []
    for r in gsm_rest:
        reasoning = r["answer"].split("####")[0].strip()
        exemplars.append((r["question"], reasoning, r["gold"]))

    rows: list[dict] = []

    # ---- source A: original GSM8K train rationales (human-written, on-distribution)
    for _ in range(args.gsm_orig_repeat):
        for r in gsm_rest:
            reasoning = r["answer"].split("####")[0].strip()
            if not clean_ok(reasoning):
                continue
            rows.append(
                {
                    "question": r["question"],
                    "solution": f"{reasoning}\n\nANSWER: {r['gold']}",
                    "src": "gsm8k_train_orig",
                }
            )

    # ---- source B: OpenMathInstruct-2
    import pyarrow.parquet as pq

    tbl = pq.read_table(
        args.omi2, columns=["problem", "generated_solution", "expected_answer", "problem_source"]
    )
    n = tbl.num_rows
    print("omi2 rows:", n)
    src_col = tbl.column("problem_source").to_pylist()
    print(Counter(src_col))

    prob = tbl.column("problem").to_pylist()
    sol = tbl.column("generated_solution").to_pylist()
    exp = tbl.column("expected_answer").to_pylist()

    idx_gsm = [i for i, s in enumerate(src_col) if "gsm8k" in s]
    idx_math = [i for i, s in enumerate(src_col) if "gsm8k" not in s]
    rng.shuffle(idx_gsm)
    rng.shuffle(idx_math)

    per_problem: Counter = Counter()
    seen_sol: set[str] = set()

    def take(idxs, budget, tag):
        kept = 0
        for i in idxs:
            if kept >= budget:
                break
            a = norm_answer(str(exp[i]))
            if a is None:
                continue
            body = unbox(sol[i])
            if body is None or len(body) < 20 or not clean_ok(body):
                continue
            if len(body) > 4000:
                continue
            key = prob[i]
            if per_problem[key] >= args.max_per_problem:
                continue
            h = hash((key, body))
            if h in seen_sol:
                continue
            seen_sol.add(h)
            per_problem[key] += 1
            rows.append(
                {"question": prob[i], "solution": f"{body}\n\nANSWER: {a}", "src": tag}
            )
            kept += 1
        print(f"{tag}: kept {kept}")

    take(idx_gsm, args.n_omi_gsm, "omi2_gsm8k")
    take(idx_math, args.n_omi_math, "omi2_math")

    # ---- render prompts, with a minority carrying the harness's 10-shot prefix
    rng.shuffle(rows)
    out = []
    n_excluded = 0
    for r in rows:
        user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip())
        if excluded and (user + "\x00" + r["solution"].strip() + "<end_of_turn>") in excluded:
            n_excluded += 1
            continue
        if rng.random() < args.fewshot_frac:
            shots = rng.sample(exemplars, 10)
            system = "\n\n".join(fewshot_block(*s) for s in shots)
            nshot = 10
        else:
            system = None
            nshot = 0
        out.append(
            {
                "system": system,
                "user": user,
                # the terminator is written into the data, not appended by the
                # trainer, so the file itself is checkable (pitfalls: eos_mismatch)
                "target": r["solution"].strip() + "<end_of_turn>",
                "src": r["src"],
                "nshot": nshot,
            }
        )

    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("dropped as already used:", n_excluded)
    print("wrote", len(out), "->", args.out)
    print(Counter(o["src"] for o in out))
    print("fewshot rows:", sum(1 for o in out if o["nshot"]))


if __name__ == "__main__":
    main()
