#!/usr/bin/env python3
"""Rejection sampling: draw k completions per train-side problem from a checkpoint,
keep the ones whose ANSWER line matches the reference answer.

Prompts are rendered with the grader's chat template, exactly as in build_data.py,
so the samples are on-policy for the distribution the grader will use.
"""
from __future__ import annotations
import argparse, json, random, re, sys
from collections import Counter, defaultdict

from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
STOP = "<end_of_turn>"
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def norm(s):
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return f"{f:.5g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True, help="jsonl of {question, answer}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-problems", type=int, default=0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template = open("templates/gemma3.jinja").read()

    pool = [json.loads(l) for l in open(args.pool)]
    if args.n_problems and args.n_problems < len(pool):
        rng.shuffle(pool)
        pool = pool[: args.n_problems]
    print(f"{len(pool)} problems x k={args.k}", flush=True)

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": MATH_PROMPT_TEMPLATE.replace("{prompt}", p["question"].strip())}],
        tokenize=False, add_generation_prompt=True) for p in pool]

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              max_model_len=2048, dtype="bfloat16", enforce_eager=False, seed=args.seed)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept, stats = [], Counter()
    per_problem_correct = []
    for p, o in zip(pool, outs):
        gold = norm(p["answer"])
        cands = []
        for c in o.outputs:
            txt = c.text
            m = re.findall(r"ANSWER:\s*([^\n]*)", txt)
            if not m:
                stats["no_marker"] += 1
                continue
            if norm(m[-1]) is None or norm(m[-1]) != gold:
                stats["wrong"] += 1
                continue
            if len(m) != 1:
                stats["multi_marker"] += 1
                continue
            body = txt.strip()
            if not body.endswith(f"ANSWER: {m[0].strip()}"):
                stats["trailing_text"] += 1
                continue
            cands.append(body)
        per_problem_correct.append(len(cands))
        # dedup then keep the shortest few: short correct chains carry less noise
        seen, uniq = set(), []
        for c in sorted(cands, key=len):
            key = re.sub(r"\s+", " ", c.lower())[:200]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        for c in uniq[: args.keep_per_problem]:
            kept.append({"question": p["question"].strip(), "completion": c + STOP,
                         "src": "rft_self"})
            stats["kept"] += 1

    solved = sum(1 for c in per_problem_correct if c > 0)
    print(f"stats: {dict(stats)}", flush=True)
    print(f"problems with >=1 correct sample: {solved}/{len(pool)} = {solved/len(pool):.3f}", flush=True)
    print(f"pass@1 (mean correct/k): {sum(per_problem_correct)/(len(pool)*args.k):.3f}", flush=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(kept)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
