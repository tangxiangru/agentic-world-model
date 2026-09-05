#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data: sample k solutions per problem from a checkpoint,
keep the ones whose final ANSWER equals the gold answer.

Problems come from GSM8K *train* and OpenMathInstruct-2's gsm8k/augmented_gsm8k problems
(both train-derived). The GSM8K test split is never touched.
"""
import argparse, json, os, random, re, sys

from datasets import load_dataset, load_from_disk

TASK = "/home/ben/task"
sys.path.insert(0, f"{TASK}/scripts")
from build_sft_data import MATH_PROMPT_TEMPLATE, clean_int  # noqa: E402

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def load_problems(n, seed):
    probs = {}
    for r in load_dataset("openai/gsm8k", "main", split="train"):
        a = clean_int(r["answer"].rpartition("####")[2])
        if a is not None:
            probs[r["question"].strip()] = a
    for r in load_from_disk(f"{TASK}/data/omi2_gsm8k_raw"):
        a = clean_int(r["expected_answer"])
        if a is not None:
            probs.setdefault(r["problem"].strip(), a)
    items = sorted(probs.items())
    random.Random(seed).shuffle(items)
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-problems", type=int, default=24000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=f"{TASK}/data/rft.jsonl")
    ap.add_argument("--stats-out", default=f"{TASK}/analysis/rft_stats.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(f"{TASK}/templates/gemma3.jinja").read()

    items = load_problems(args.n_problems, args.seed)
    print(f"{len(items)} problems", flush=True)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
            tokenize=False, add_generation_prompt=True)
        for q, _ in items
    ]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048, enable_prefix_caching=True, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept, n_correct, n_total, solved = [], 0, 0, 0
    for (q, gold), o in zip(items, outs):
        good, seen = [], set()
        for c in o.outputs:
            n_total += 1
            t = c.text.strip()
            m = ANS_RE.search(t)
            if not m or clean_int(m.group(1)) != gold:
                continue
            n_correct += 1
            key = re.sub(r"\s+", " ", t)
            if key in seen:
                continue
            seen.add(key)
            good.append(t)
        if good:
            solved += 1
        good.sort(key=len)  # prefer the shortest correct chains
        for t in good[: args.keep_per_problem]:
            kept.append({"system": None,
                         "user": MATH_PROMPT_TEMPLATE.format(prompt=q),
                         "target": t + "<end_of_turn>",
                         "answer": gold, "src": "rft",
                         "text": q + "\n\n" + t})
    random.Random(args.seed).shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {"problems": len(items), "samples": n_total, "correct": n_correct,
             "sample_accuracy": round(n_correct / max(n_total, 1), 4),
             "problems_solved_at_least_once": solved,
             "pass_at_k": round(solved / max(len(items), 1), 4), "kept_rows": len(kept)}
    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
    json.dump(stats, open(args.stats_out, "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
