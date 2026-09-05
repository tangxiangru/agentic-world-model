#!/usr/bin/env python3
"""Build the SFT mixture for GSM8K.

Every row is rendered with the *exact* chat template the grader uses
(templates/gemma3.jinja, hash-checked) so training and grading see the same
string.  Targets always end with the eval's answer marker on its own last line
and then the template terminator <end_of_turn>.

Sources (none of them the GSM8K test split):
  * openai/gsm8k, split=train              -- human gold chains
  * nvidia/OpenMathInstruct-2, gsm8k rows  -- model-written chains, answer-verified
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

TASK = "/home/ben/task"
TEMPLATE_PATH = os.path.join(TASK, "templates", "gemma3.jinja")
SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

# copied verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANSWER_MARKER = "ANSWER: "
STOP_TOKEN = "<end_of_turn>"

CALC = re.compile(r"<<[^>]*>>")
BOXED = re.compile(r"\\boxed\{")


def strip_calc(text: str) -> str:
    return CALC.sub("", text)


def clean_number(s: str) -> str | None:
    """GSM8K targets are plain integers; keep only rows whose answer is one."""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if re.fullmatch(r"-?\d+", s):
        return str(int(s))
    if re.fullmatch(r"-?\d+\.0+", s):
        return str(int(float(s)))
    return None


def drop_boxed_tail(sol: str) -> str:
    """Remove the trailing '\\boxed{..}' sentence so only one answer marker remains."""
    i = sol.rfind("\\boxed{")
    if i == -1:
        return sol.strip()
    # cut back to the start of the sentence/line that holds the box
    j = sol.rfind("\n", 0, i)
    head = sol[: j if j != -1 else i]
    return head.strip()


def looks_clean(sol: str) -> bool:
    if not sol or len(sol) < 20:
        return False
    if "\\boxed" in sol or "####" in sol or "ANSWER:" in sol:
        return False
    if "```" in sol or "def " in sol or "print(" in sol:
        return False
    return True


# --------------------------------------------------------------------------


def load_gsm8k_train():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        q = r["question"].strip()
        body, _, ans = r["answer"].partition("####")
        a = clean_number(ans)
        if a is None:
            continue
        sol = strip_calc(body).strip()
        sol = "\n".join(line.rstrip() for line in sol.splitlines() if line.strip())
        if not looks_clean(sol):
            continue
        out.append({"question": q, "solution": sol, "answer": a, "src": "gsm8k_gold"})
    return out


def load_openmath(n_orig: int, n_aug: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    keep_sources = {"gsm8k", "augmented_gsm8k"}
    ds = ds.filter(
        lambda b: [s in keep_sources for s in b["problem_source"]],
        batched=True,
        num_proc=16,
    )
    print("openmath gsm8k rows:", len(ds), flush=True)
    orig, aug = [], []
    for r in ds:
        a = clean_number(r["expected_answer"])
        if a is None:
            continue
        sol = drop_boxed_tail(r["generated_solution"])
        if not looks_clean(sol):
            continue
        rec = {
            "question": r["problem"].strip(),
            "solution": sol,
            "answer": a,
            "src": r["problem_source"],
        }
        (orig if r["problem_source"] == "gsm8k" else aug).append(rec)
    rng = random.Random(seed)
    rng.shuffle(orig)
    rng.shuffle(aug)
    return orig[:n_orig], aug[:n_aug]


# --------------------------------------------------------------------------


def render(tok, question: str, solution: str, answer: str, shots: list) -> dict:
    msgs = []
    if shots:
        sysmsg = "\n\n".join(
            f"{s['question']}\n\nReasoning:\n{s['solution']}\n\n{ANSWER_MARKER}{s['answer']}"
            for s in shots
        )
        msgs.append({"role": "system", "content": sysmsg})
    msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=question)})
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    target = f"{solution}\n\n{ANSWER_MARKER}{answer}{STOP_TOKEN}"
    return {"prompt": prompt, "completion": target}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(TASK, "data", "sft_mix.jsonl"))
    ap.add_argument("--n-orig", type=int, default=7473)
    ap.add_argument("--n-aug", type=int, default=45000)
    ap.add_argument("--gold-repeat", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--max-tokens", type=int, default=1408)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    with open(TEMPLATE_PATH) as f:
        template = f.read()
    tok.chat_template = template
    print("template sha256:", hashlib.sha256(template.encode()).hexdigest(), flush=True)

    gold = load_gsm8k_train()
    print("gsm8k gold:", len(gold), flush=True)
    om_orig, om_aug = load_openmath(args.n_orig, args.n_aug, args.seed)
    print("openmath orig/aug kept:", len(om_orig), len(om_aug), flush=True)

    pool = []
    for _ in range(args.gold_repeat):
        pool += gold
    pool += om_orig + om_aug

    # dedup on (question, solution)
    seen = set()
    uniq = []
    for r in pool:
        k = hashlib.sha1((r["question"] + "||" + r["solution"]).encode()).hexdigest()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print("after dedup:", len(uniq), flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    shot_pool = gold  # few-shot demos always come from human gold chains

    rows, lens, dropped = [], [], 0
    for r in uniq:
        shots = []
        if rng.random() < args.fewshot_frac:
            k = rng.choice([2, 3, 4])
            shots = rng.sample(shot_pool, k)
        row = render(tok, r["question"], r["solution"], r["answer"], shots)
        n_p = len(tok(row["prompt"], add_special_tokens=False)["input_ids"])
        n_c = len(tok(row["completion"], add_special_tokens=False)["input_ids"])
        if n_p + n_c > args.max_tokens:
            dropped += 1
            continue
        row["src"] = r["src"]
        row["n_tokens"] = n_p + n_c
        rows.append(row)
        lens.append(n_p + n_c)

    lens.sort()
    print(f"rows={len(rows)} dropped_too_long={dropped}", flush=True)
    print(
        "tokens p50=%d p90=%d p99=%d max=%d"
        % (lens[len(lens) // 2], lens[int(0.9 * len(lens))], lens[int(0.99 * len(lens))], lens[-1]),
        flush=True,
    )
    from collections import Counter

    print(Counter(r["src"] for r in rows), flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", args.out, flush=True)

    # plain-text dump for the contamination checker
    with open(args.out.replace(".jsonl", ".contam.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["prompt"] + r["completion"]}) + "\n")


if __name__ == "__main__":
    main()
