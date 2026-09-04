"""Build the SFT corpus for GSM8K, rendered exactly the way the grader renders.

Sources (all GSM8K *train* / MATH *train* derived - never the test split):
  * nvidia/OpenMathInstruct-2 train_1M, problem_source in {gsm8k, augmented_gsm8k}
  * openai/gsm8k main/train (7473) - the native terse style the 10-shot prompt shows

Output: jsonl with {"prompt", "completion", "answer", "src"} where
  prompt     = the full rendered chat prefix up to '<start_of_turn>model\n'
  completion = chain of thought + 'ANSWER: N' + '<end_of_turn>'
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SNAPSHOT = os.environ["PTB_BASE_MODEL_SNAPSHOT"]


def render_prompt(tok, question: str, shots=None) -> str:
    msgs = []
    if shots:
        msgs.append({"role": "system", "content": fmt.fewshot_block(shots)})
    msgs.append({"role": "user", "content": fmt.user_prompt(question)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


N_GRAM = 10


def shingles(text: str) -> set:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + N_GRAM]) for i in range(max(0, len(w) - N_GRAM + 1))}


def is_holdout(q: str, hold_q: set, hold_shingles: set) -> bool:
    if q in hold_q:
        return True
    return bool(shingles(q) & hold_shingles)


def clean_omi2(sol: str, ans: str) -> str | None:
    s = fmt.delatex(sol)
    # drop a trailing "The final answer is N." style line; we append our own marker
    s = re.sub(r"\n*(?:The (?:final )?answer is[^\n]*)$", "", s).strip()
    if not s or len(s) < 20:
        return None
    if "ANSWER:" in s or "\\boxed" in s or "####" in s:
        return None
    return s


def clean_gsm8k(sol: str) -> str | None:
    body = sol.split("####")[0]
    body = fmt.strip_gsm8k_calc(body).strip()
    body = re.sub(r"[ \t]+", " ", body)
    if not body:
        return None
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi2", type=int, default=60000)
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default=None,
                    help="jsonl of held-out dev questions to exclude from training")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = fmt.load_template()

    # --- private dev holdout: exact questions plus any 10-gram-overlapping rewrite ----
    hold_q, hold_shingles = set(), set()
    if args.holdout:
        for line in open(args.holdout):
            q = json.loads(line)["question"]
            hold_q.add(q)
            hold_shingles |= shingles(q)

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    # few-shot pool: the grader draws its 10 shots from this same split
    pool = []
    for r in gsm.select(range(400)):
        if r["question"] in hold_q:
            continue
        a = fmt.normalize_answer(r["answer"].split("####")[-1])
        b = clean_gsm8k(r["answer"])
        if a and b:
            pool.append((r["question"], b, a))

    rows = []

    # ---- 1. OpenMathInstruct-2, gsm8k-flavoured -----------------------------
    omi = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    keep_src = {"gsm8k", "augmented_gsm8k"}
    idx = [i for i, s in enumerate(omi["problem_source"]) if s in keep_src]
    rng.shuffle(idx)
    per_problem: dict[str, int] = {}
    n_taken = n_holdout_blocked = 0
    for i in idx:
        if n_taken >= args.n_omi2:
            break
        r = omi[i]
        ans = fmt.normalize_answer(str(r["expected_answer"]))
        if ans is None:
            continue
        key = r["problem"]
        if per_problem.get(key, 0) >= args.max_per_problem:
            continue
        if is_holdout(key, hold_q, hold_shingles):
            n_holdout_blocked += 1
            continue
        sol = clean_omi2(r["generated_solution"], ans)
        if sol is None:
            continue
        per_problem[key] = per_problem.get(key, 0) + 1
        rows.append({"q": key, "sol": sol, "ans": ans, "src": r["problem_source"]})
        n_taken += 1

    # ---- 2. native GSM8K train ---------------------------------------------
    native = []
    for r in gsm:
        if r["question"] in hold_q:
            n_holdout_blocked += 1
            continue
        ans = fmt.normalize_answer(r["answer"].split("####")[-1])
        sol = clean_gsm8k(r["answer"])
        if ans is None or sol is None:
            continue
        native.append({"q": r["question"], "sol": sol, "ans": ans, "src": "gsm8k_native"})
    for _ in range(args.gsm8k_repeat):
        rows.extend(native)

    rng.shuffle(rows)

    # ---- 3. render ----------------------------------------------------------
    with open(args.out, "w") as f:
        for r in rows:
            shots = None
            if rng.random() < args.fewshot_frac:
                k = rng.choice([2, 3, 4, 10])
                cand = [p for p in pool if p[0] != r["q"]]
                shots = rng.sample(cand, k)
            prompt = render_prompt(tok, r["q"], shots)
            completion = fmt.build_target(r["sol"], r["ans"]) + fmt.STOP_TOKEN
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                "answer": r["ans"],
                "src": r["src"],
                "question": r["q"],
            }) + "\n")

    print(f"wrote {len(rows)} rows to {args.out}; holdout-blocked {n_holdout_blocked}")


if __name__ == "__main__":
    main()
