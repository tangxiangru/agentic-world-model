#!/usr/bin/env python3
"""Build SFT data for gemma-3-4b-pt -> GSM8K, in the grader's exact prompt format.

The grader (inspect_evals/gsm8k via evaluate.py) builds, for every test item:

  system  = "\n\n".join(sample_to_fewshot(s) for s in 10 gsm8k TRAIN items)   # fewshot_seed=42
  user    = MATH_PROMPT_TEMPLATE.format(prompt=question)
  render  = templates/gemma3.jinja, add_generation_prompt=True

and scores with match(numeric=True, location="end") -> the LAST numeric word of
the completion must equal the gold answer.

So a training row is
  prompt     = <the same render>
  completion = "<chain of thought>\n\nANSWER: <n><end_of_turn>"

Sources (all GSM8K *train*-derived or independently authored; never the test split):
  * openai/gsm8k          split=train        (7473) - original human solutions
  * meta-math/MetaMathQA  GSM_* subsets              - augmentations of gsm8k TRAIN
  * nvidia/OpenMathInstruct-2  problem_source in {gsm8k, augmented_gsm8k}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

from datasets import load_from_disk  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"

# byte-for-byte copy of inspect_evals.gsm8k.MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP = "<end_of_turn>"
CALC = re.compile(r"<<[^>]*>>")
NUMLIKE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


# --------------------------------------------------------------------------- #
# answer normalisation
# --------------------------------------------------------------------------- #
def clean_number(s: str) -> str | None:
    """Return a canonical plain-number string, or None if it is not a number."""
    s = s.strip().rstrip(".").replace("$", "").replace(",", "").replace("%", "")
    s = s.replace("\\!", "").replace("\\,", "").strip()
    if not NUMLIKE.match(s.replace(",", "")):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or None


def strip_boxed(text: str) -> str:
    """Remove a trailing \\boxed{...} sentence so only one answer marker remains."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return text
    # cut back to the start of the sentence/line that holds the box
    start = max(text.rfind("\n", 0, idx), text.rfind(". ", 0, idx) + 1)
    return text[: start if start > 0 else idx].rstrip()


def last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return clean_number(nums[-1]) if nums else None


# --------------------------------------------------------------------------- #
# source readers -> list of (question, reasoning, answer)
# --------------------------------------------------------------------------- #
def from_gsm8k_train(path: str) -> list[tuple[str, str, str]]:
    ds = load_from_disk(path)["train"]
    out = []
    for q, a in zip(ds["question"], ds["answer"]):
        body, _, ans = a.partition("####")
        ans = clean_number(ans)
        if ans is None:
            continue
        body = CALC.sub("", body).strip()
        out.append((q.strip(), body, ans))
    return out


def from_metamath(path: str, types: set[str], cap_per_type: int, offset: int = 0) -> list[tuple[str, str, str]]:
    ds = load_from_disk(path)
    out, seen = [], {t: 0 for t in types}
    for t, q, r in zip(ds["type"], ds["query"], ds["response"]):
        if t not in types:
            continue
        seen[t] += 1
        if seen[t] <= offset or seen[t] > offset + cap_per_type:
            continue
        body, sep, ans = r.rpartition("The answer is:")
        if not sep:
            continue
        ans = clean_number(ans)
        if ans is None:
            continue
        body = strip_boxed(body.strip())
        if len(body) < 30:
            continue
        out.append((q.strip(), body, ans))
    return out


def from_omi2(path: str, sources: set[str], cap: int, seed: int, offset: int = 0) -> list[tuple[str, str, str]]:
    ds = load_from_disk(path)
    keep = ds.filter(lambda r: r["problem_source"] in sources, num_proc=8)
    keep = keep.shuffle(seed=seed)
    lo = min(offset, len(keep))
    keep = keep.select(range(lo, min(lo + cap * 2, len(keep))))
    out = []
    for q, sol, ans in zip(keep["problem"], keep["generated_solution"], keep["expected_answer"]):
        a = clean_number(ans)
        if a is None:
            continue
        body = strip_boxed(sol.strip())
        if len(body) < 30:
            continue
        out.append((q.strip(), body, a))
        if len(out) >= cap:
            break
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.30,
                    help="fraction of rows that carry the grader's exact 10-shot system message")
    ap.add_argument("--varshot-frac", type=float, default=0.15,
                    help="fraction of rows carrying k in {2,4,6,8} demos from a disjoint train pool")
    ap.add_argument("--gsm8k-repeat", type=int, default=2)
    ap.add_argument("--metamath-cap", type=int, default=12000)
    ap.add_argument("--omi2-cap", type=int, default=40000)
    ap.add_argument("--omi2-offset", type=int, default=0)
    ap.add_argument("--metamath-offset", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open(TEMPLATE).read()

    # --- prompt-context pools (gsm8k TRAIN only) ---------------------------- #
    eval_sys = open("/home/ben/task/data/eval_system_message.txt").read()
    eval_ids = {json.loads(l)["id"] for l in open("/home/ben/task/data/eval_fewshot_pool.jsonl")}

    gsm_train = from_gsm8k_train("/home/ben/task/data/gsm8k_raw")
    demo_pool = [
        f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}"
        for q, r, a in gsm_train[:400]
    ]

    # --- solution sources --------------------------------------------------- #
    rows: list[dict[str, Any]] = []
    src_counts: dict[str, int] = {}

    def add(items, src):
        for q, r, a in items:
            rows.append({"question": q, "reasoning": r, "answer": a, "src": src})
        src_counts[src] = src_counts.get(src, 0) + len(items)

    for _ in range(args.gsm8k_repeat):
        add(gsm_train, "gsm8k_train")
    add(from_metamath("/home/ben/task/data/metamathqa",
                      {"GSM_AnsAug", "GSM_Rephrased"}, args.metamath_cap,
                      args.metamath_offset), "metamath_gsm")
    if os.path.isdir("/home/ben/task/data/omi2_1M"):
        add(from_omi2("/home/ben/task/data/omi2_1M",
                      {"gsm8k", "augmented_gsm8k"}, args.omi2_cap, args.seed,
                      args.omi2_offset), "omi2_gsm")

    # dedup identical (question, solution) inside each augmented source. gsm8k_train is
    # deliberately repeated --gsm8k-repeat times as an up-weighting, so it is exempt.
    seen_key, deduped = set(), []
    for r in rows:
        if r["src"] == "gsm8k_train":
            deduped.append(r)
            continue
        k = (r["question"], r["reasoning"])
        if k in seen_key:
            continue
        seen_key.add(k)
        deduped.append(r)
    print(json.dumps({"before_dedup": len(rows), "after_dedup": len(deduped)}))
    rows = deduped

    rng.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]

    # --- render -------------------------------------------------------------- #
    n_fs, n_vs, lens = 0, 0, []
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            u = rng.random()
            if u < args.fewshot_frac:
                system, kind = eval_sys, "eval10"
                n_fs += 1
            elif u < args.fewshot_frac + args.varshot_frac:
                k = rng.choice([2, 4, 6, 8])
                system = "\n\n".join(rng.sample(demo_pool, k))
                kind = f"var{k}"
                n_vs += 1
            else:
                system, kind = None, "zero"

            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])})
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            body = r["reasoning"].strip()
            # exactly one "ANSWER:" marker, on its own final line, then the stop token
            body = re.sub(r"(?im)^\s*answer\s*:.*$", "", body).strip()
            completion = f"{body}\n\nANSWER: {r['answer']}{STOP}"

            n_p = len(tok(prompt, add_special_tokens=False)["input_ids"])
            n_c = len(tok(completion, add_special_tokens=False)["input_ids"])
            lens.append(n_p + n_c)
            f.write(json.dumps({
                "prompt": prompt, "completion": completion, "target": completion,
                "answer": r["answer"], "src": r["src"], "ctx": kind,
                "n_prompt": n_p, "n_completion": n_c,
            }) + "\n")

    lens.sort()
    print(json.dumps({
        "rows": len(rows), "sources": src_counts,
        "ctx": {"eval10": n_fs, "var": n_vs, "zero": len(rows) - n_fs - n_vs},
        "tok_p50": lens[len(lens) // 2], "tok_p95": lens[int(len(lens) * 0.95)],
        "tok_p99": lens[int(len(lens) * 0.99)], "tok_max": lens[-1],
        "tok_total_millions": round(sum(lens) / 1e6, 2),
        "eval_fewshot_ids_in_pool": len(eval_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
