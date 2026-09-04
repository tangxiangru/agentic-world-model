#!/usr/bin/env python3
"""Build SFT data rendered exactly the way templates/gemma3.jinja renders it for the grader.

Row format written to jsonl:
  {"id":..., "source":..., "prompt": <text up to and including '<start_of_turn>model\\n'>,
   "target": <model turn content + '<end_of_turn>'>, "gold": "<numeric answer>"}

The trainer tokenises prompt+target and masks the prompt, so `target` is the only
text that carries loss. `target` ends with the terminator the grading template stops on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

# byte-for-byte copy of inspect_evals.gsm8k.gsm8k.MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"


def render_prompt(question: str, fewshot_block: str | None = None) -> str:
    """Reproduce templates/gemma3.jinja for [system?, user] with add_generation_prompt=True."""
    user = MATH_PROMPT_TEMPLATE.format(prompt=question).strip()
    prefix = (fewshot_block + "\n\n") if fewshot_block else ""
    return f"{BOS}{SOT}user\n{prefix}{user}{EOT}\n{SOT}model\n"


def fewshot_text(question: str, reasoning: str, gold: str) -> str:
    """Reproduce inspect_evals.gsm8k.sample_to_fewshot."""
    return f"{question}\n\nReasoning:\n{reasoning}\n\nANSWER: {gold}"


def sid(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- gsm8k train
def load_gsm8k_train():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        q = r["question"].strip()
        parts = r["answer"].split("####")
        reasoning = "####".join(parts[:-1]).strip()
        gold = parts[-1].strip().replace(",", "")
        out.append({"question": q, "reasoning": reasoning, "gold": gold})
    return out


# ------------------------------------------------------- OpenMathInstruct-2
BOXED_RE = re.compile(r"\\boxed\s*\{")


def strip_boxed_tail(sol: str) -> str | None:
    """Remove the final 'The final answer is $\\boxed{...}$.' sentence; keep the rest."""
    i = sol.rfind("\\boxed")
    if i == -1:
        return None
    # walk back to the start of the sentence/line containing the box
    j = sol.rfind("\n", 0, i)
    j = j if j != -1 else 0
    head = sol[:j].rstrip()
    if not head:
        return None
    return head


def load_omi2(n_gsm8k: int, n_math: int, seed: int):
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    gsrc = {"gsm8k", "augmented_gsm8k"}
    msrc = {"math", "augmented_math"}
    keep = gsrc | msrc
    ds = ds.filter(
        lambda b: [s in keep for s in b["problem_source"]], batched=True, num_proc=12
    ).shuffle(seed=seed)
    got_g, got_m, out = 0, 0, []
    for r in ds:
        src = r["problem_source"]
        is_g = src in gsrc
        is_m = src in msrc
        if is_g and got_g >= n_gsm8k:
            if got_m >= n_math:
                break
            continue
        if is_m and got_m >= n_math:
            continue
        ans = str(r["expected_answer"]).strip()
        # the grader reads the LAST number in the completion: only keep plain numeric answers
        if not re.fullmatch(r"-?\d+(\.\d+)?", ans.replace(",", "")):
            continue
        body = strip_boxed_tail(r["generated_solution"])
        if body is None or BOXED_RE.search(body):
            continue
        out.append(
            {
                "question": r["problem"].strip(),
                "reasoning": body,
                "gold": ans.replace(",", ""),
                "source": src,
            }
        )
        if is_g:
            got_g += 1
        else:
            got_m += 1
        if got_g >= n_gsm8k and got_m >= n_math:
            break
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-omi-gsm8k", type=int, default=0)
    ap.add_argument("--n-omi-math", type=int, default=0)
    ap.add_argument("--gsm8k-repeat", type=int, default=1)
    ap.add_argument("--fewshot-frac", type=float, default=0.05)
    ap.add_argument("--fewshot-max-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gsm = load_gsm8k_train()
    print(f"gsm8k train rows: {len(gsm)}")

    pool = []
    for _ in range(args.gsm8k_repeat):
        for r in gsm:
            pool.append({**r, "source": "gsm8k_train"})
    if args.n_omi_gsm8k or args.n_omi_math:
        omi = load_omi2(args.n_omi_gsm8k, args.n_omi_math, args.seed)
        print(f"omi2 rows kept: {len(omi)}")
        pool += omi

    # dedup on (question, gold, reasoning)
    seen, dedup = set(), []
    for r in pool:
        k = sid(r["question"] + "|" + r["reasoning"] + "|" + r["gold"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    print(f"after dedup: {len(dedup)}")
    rng.shuffle(dedup)

    # few-shot blocks are drawn from gsm8k TRAIN only
    fs_pool = gsm

    n_fs = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in dedup:
            block = None
            if rng.random() < args.fewshot_frac:
                k = rng.randint(1, args.fewshot_max_k)
                shots = rng.sample(fs_pool, k)
                shots = [s for s in shots if s["question"] != r["question"]]
                if shots:
                    block = "\n\n".join(
                        fewshot_text(s["question"], s["reasoning"], s["gold"])
                        for s in shots
                    )
                    n_fs += 1
            prompt = render_prompt(r["question"], block)
            target = f"{r['reasoning'].strip()}\n\nANSWER: {r['gold']}{EOT}"
            f.write(
                json.dumps(
                    {
                        "id": sid(prompt + target),
                        "source": r["source"],
                        "prompt": prompt,
                        "target": target,
                        "gold": r["gold"],
                    }
                )
                + "\n"
            )
    print(f"wrote {out_path} rows={len(dedup)} fewshot_rows={n_fs}")


if __name__ == "__main__":
    main()
