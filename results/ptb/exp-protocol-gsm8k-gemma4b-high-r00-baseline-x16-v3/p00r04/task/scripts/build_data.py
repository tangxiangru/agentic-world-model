"""Build the SFT corpus in the exact string format the grader renders.

Sources
  * nvidia/OpenMathInstruct-2, rows whose problem_source is gsm8k / augmented_gsm8k
    (problems are GSM8K *train* items or LLM-generated variants of them; solutions
    by Llama-3.1-405B-Instruct, answer-verified by the dataset authors).
  * openai/gsm8k train split, minus the 250 items held out in data/probe250.jsonl.

Every target is `<reasoning>\n\nANSWER: <n><end_of_turn>` so that
  * the answer marker appears exactly once, and
  * the last whitespace-separated numeric token of the completion is the answer,
    which is what inspect's match(location="end", numeric=True) reads.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import fmt as F  # noqa: E402

NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def clean_answer(a: str) -> str | None:
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUM_RE.match(a):
        return None
    if a.endswith(".0"):
        a = a[:-2]
    return a


BOXED_TAIL = re.compile(
    r"\s*(?:so\s+)?(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?\\boxed\{[^}]*\}\$?\.?\s*$",
    re.IGNORECASE,
)
BOXED_ANY_TAIL = re.compile(r"\s*\$?\\boxed\{[^}]*\}\$?\.?\s*$")


def strip_boxed_tail(sol: str) -> str:
    s = sol.strip()
    s2 = BOXED_TAIL.sub("", s)
    if s2 != s:
        return s2.strip()
    return BOXED_ANY_TAIL.sub("", s).strip()


def sanitize(body: str) -> str:
    """Remove leftover LaTeX boxes anywhere in the body; keep it plain text."""
    body = re.sub(r"\\boxed\{([^}]*)\}", r"\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def last_number(text: str) -> str | None:
    words = re.split(r"\s+", text.strip())
    for w in reversed(words):
        w2 = w.strip().strip(".,:;!?$()[]").replace(",", "")
        if w2.replace(".", "").replace("-", "").isnumeric():
            return w2
    return None


def build_omi(path: str, max_per_problem: int, cap: int, rng: random.Random):
    """Original-GSM8K-problem rows first (highest quality), augmented ones to fill."""
    per = defaultdict(list)
    src_of = {}
    n_read = 0
    with open(path) as f:
        for line in f:
            n_read += 1
            r = json.loads(line)
            ans = clean_answer(r["expected_answer"])
            if ans is None:
                continue
            body = sanitize(strip_boxed_tail(r["generated_solution"]))
            if not body or len(body) < 30 or len(body) > 3500:
                continue
            # reject bodies that still look like they carry a second answer marker
            if "ANSWER:" in body or "####" in body:
                continue
            per[r["problem"]].append((body, ans))
            src_of[r["problem"]] = r["problem_source"]
    print(f"  read {n_read} omi rows -> {len(per)} unique problems", flush=True)
    tiers = {"gsm8k": [], "augmented_gsm8k": []}
    for prob, sols in per.items():
        rng.shuffle(sols)
        k = max_per_problem if src_of[prob] == "gsm8k" else max(1, max_per_problem - 1)
        for body, ans in sols[:k]:
            tiers[src_of[prob]].append(
                {"question": prob, "body": body, "answer": ans,
                 "src": "omi2_" + src_of[prob]}
            )
    for v in tiers.values():
        rng.shuffle(v)
    out = tiers["gsm8k"][:cap]
    out += tiers["augmented_gsm8k"][: max(0, cap - len(out))]
    rng.shuffle(out)
    return out


def build_gsm8k_train(path: str):
    out = []
    for line in open(path):
        r = json.loads(line)
        sol, ans = r["answer"].rsplit("####", 1)
        ans = clean_answer(ans)
        if ans is None:
            continue
        # drop the calculator annotations: they are noise the grader never needs
        body = re.sub(r"<<[^>]*>>", "", sol).strip()
        out.append({"question": r["question"], "body": body, "answer": ans, "src": "gsm8k_train"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omi", default="data/omi2_gsm8k_1M.jsonl")
    ap.add_argument("--gsm8k-pool", default="data/gsm8k_train_pool.jsonl")
    ap.add_argument("--omi-cap", type=int, default=60000)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--decon-out", default="data/sft_train_decon_input.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    if os.path.exists(args.omi):
        rows += build_omi(args.omi, args.max_per_problem, args.omi_cap, rng)
    g = build_gsm8k_train(args.gsm8k_pool)
    for _ in range(args.gsm8k_repeat):
        rows += list(g)
    rng.shuffle(rows)

    n_bad = 0
    out = []
    for i, r in enumerate(rows):
        target = F.render_target(r["body"], r["answer"])
        # invariant 1: exactly one answer marker
        if target.count(F.ANSWER_MARKER) != 1:
            n_bad += 1
            continue
        # invariant 2: the last numeric token is the answer
        vis = target[: -len(F.STOP_TOKEN)]
        if last_number(vis) != r["answer"].replace(",", ""):
            n_bad += 1
            continue
        out.append(
            {
                "question": r["question"],
                "target": target,
                "answer": r["answer"],
                "src": r["src"],
                "fewshot": rng.random() < args.fewshot_frac,
            }
        )
    print(f"kept {len(out)}  dropped-by-invariant {n_bad}")

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    # contamination checker reads a 'text' field
    with open(args.decon_out, "w") as f:
        for r in out:
            f.write(json.dumps({"text": r["question"] + "\n" + r["target"]}) + "\n")
    from collections import Counter

    print(Counter(r["src"] for r in out), "fewshot rows:", sum(r["fewshot"] for r in out))
    print("wrote", args.out, args.decon_out)


if __name__ == "__main__":
    main()
