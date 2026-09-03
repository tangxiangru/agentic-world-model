#!/usr/bin/env python3
"""Rejection-sampling / STaR data generation with the current policy.

Samples k solutions per training problem with vLLM, keeps only those whose final
"ANSWER:" line matches the gold answer, dedupes, and writes a new SFT file.
Problems come from the GSM8K *train* split and from OpenMathInstruct-2's
gsm8k-derived problems (both train-derived; the test split is never touched).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import defaultdict

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

OMI_GLOB = "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/*.parquet"
ANS_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.MULTILINE)


def norm(s: str) -> str:
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


def extract(text: str) -> str | None:
    m = ANS_RE.findall(text)
    return norm(m[-1]) if m else None


def fewshot_pool(n=400):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for row in ds.select(range(n)):
        reasoning, _, final = row["answer"].partition("####")
        reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
        out.append(f"{row['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {final.strip()}")
    return out


def load_problems(n_omi: int, seed: int = 7):
    from datasets import load_dataset

    probs = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for row in ds:
        probs.append((row["question"], norm(row["answer"].split("####")[-1])))
    if n_omi > 0:
        import pyarrow.parquet as pq

        seen = set(q for q, _ in probs)
        pool = []
        for f in sorted(glob.glob(OMI_GLOB)):
            t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"]).to_pydict()
            for p, a, s in zip(t["problem"], t["expected_answer"], t["problem_source"]):
                if s == "augmented_gsm8k" and p not in seen:
                    seen.add(p)
                    pool.append((p, norm(a)))
            if len(pool) > n_omi * 3:
                break
        random.Random(seed).shuffle(pool)
        probs += pool[:n_omi]
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="work/rft.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-omi", type=int, default=25000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--fewshot-prob", type=float, default=0.15)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    probs = load_problems(args.n_omi)
    print(f"{len(probs)} problems", flush=True)

    rng = random.Random(args.seed)
    pool = fewshot_pool() if args.fewshot_prob > 0 else []

    prompts, meta = [], []
    for idx, (q, a) in enumerate(probs):
        user = MATH_PROMPT_TEMPLATE.format(prompt=q.strip())
        if pool and rng.random() < args.fewshot_prob:
            k = rng.choice([2, 3, 4, 5])
            user = "\n\n".join(rng.sample(pool, k)) + "\n\n" + user
        prompts.append(f"<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n")
        meta.append((idx, q, a, user))

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=2048,
        enable_prefix_caching=True,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<end_of_turn>"],
        seed=args.seed,
    )
    outs = llm.generate(prompts, sp)

    kept = defaultdict(list)
    n_correct = n_total = 0
    solved = 0
    for (idx, q, gold, user), o in zip(meta, outs):
        ok_any = False
        for c in o.outputs:
            n_total += 1
            txt = c.text.strip()
            if c.finish_reason != "stop":
                continue
            pred = extract(txt)
            if pred is None or pred != gold:
                continue
            n_correct += 1
            ok_any = True
            kept[idx].append((user, txt))
        solved += int(ok_any)

    print(f"sample-level correct {n_correct}/{n_total} = {n_correct/max(n_total,1):.3f}", flush=True)
    print(f"problems with >=1 correct: {solved}/{len(probs)} = {solved/len(probs):.3f}", flush=True)

    n_written = 0
    with open(args.out, "w") as f:
        for idx, sols in kept.items():
            # prefer shorter (usually cleaner) solutions, keep a couple per problem
            uniq, seen_txt = [], set()
            for user, txt in sorted(sols, key=lambda t: len(t[1])):
                key = txt[:200]
                if key in seen_txt:
                    continue
                seen_txt.add(key)
                uniq.append((user, txt))
            for user, txt in uniq[: args.keep_per_problem]:
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": user},
                                {"role": "assistant", "content": txt},
                            ],
                            "question": probs[idx][0],
                            "answer": probs[idx][1],
                        }
                    )
                    + "\n"
                )
                n_written += 1
    print(f"wrote {n_written} examples to {args.out}", flush=True)
    if args.stats_out:
        json.dump(
            {
                "sample_accuracy": n_correct / max(n_total, 1),
                "problems_solved_frac": solved / len(probs),
                "n_written": n_written,
            },
            open(args.stats_out, "w"),
            indent=2,
        )


if __name__ == "__main__":
    main()
