#!/usr/bin/env python3
"""Rejection-sampling data: sample k solutions per problem from a checkpoint,
keep the ones whose final number equals the gold answer.

Problems come from GSM8K *train* (minus the probe holdout) and from
OpenMathInstruct-2's augmented_gsm8k problems, which are LLM rewrites of GSM8K
train problems - the test split is never involved. Gold answers come with the
problems, so the filter is exact, not model-judged.

Output rows are in the same prompt/completion shape build_data.py writes, so
scripts/preflight_data.py checks them the same way.
"""
import argparse
import json
import random
import re
from collections import defaultdict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TASK = "/home/ben/task"

MATH_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:"""

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
NUMBERISH = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
EOT = "<end_of_turn>"


def last_number(text):
    for w in reversed(re.split(r"\s+", text.strip())):
        w = w.replace(",", "").rstrip(".").lstrip("$")
        if NUM.fullmatch(w):
            return w[:-2] if w.endswith(".0") else w
    return None


def norm_ans(a):
    a = a.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUMBERISH.match(a):
        return None
    return a[:-2] if a.endswith(".0") else a


def collect_problems(args, rng):
    held = {" ".join(json.loads(l)["question"].split()).lower()
            for l in open(f"{TASK}/data/probe250.jsonl")}
    probs = []
    with open(f"{TASK}/data/raw/gsm8k_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            q = d["question"].strip()
            if " ".join(q.split()).lower() in held:
                continue
            probs.append({"q": q, "a": d["answer"].split("####")[-1].strip().replace(",", ""),
                          "src": "gsm8k_train"})
    aug, seen = [], set()
    with open(f"{TASK}/data/raw/omi2_1M.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["problem_source"] != "augmented_gsm8k":
                continue
            q = d["problem"].strip()
            k = " ".join(q.split()).lower()
            if k in seen or k in held:
                continue
            seen.add(k)
            a = norm_ans(d["expected_answer"] or "")
            if a is None or len(q) > 1200:
                continue
            aug.append({"q": q, "a": a, "src": "omi2_aug"})
    rng.shuffle(aug)
    probs += aug[: args.n_augmented]
    rng.shuffle(probs)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--n-augmented", type=int, default=25000)
    ap.add_argument("--max-keep-per-problem", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    probs = collect_problems(args, rng)
    print(f"{len(probs)} problems x k={args.k}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tpl = open(f"{TASK}/templates/gemma3.jinja").read()
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=p["q"])}],
        chat_template=tpl, tokenize=False, add_generation_prompt=True) for p in probs]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    kept, n_gen, n_correct = [], 0, 0
    solved = defaultdict(int)
    per_problem_correct = []
    for p, prompt, o in zip(probs, prompts, outs):
        cands, seen_txt = [], set()
        for c in o.outputs:
            n_gen += 1
            txt = c.text
            if c.finish_reason != "stop":
                continue          # hit the cap: the target would have no terminator
            if last_number(txt) != p["a"]:
                continue
            if txt.count("ANSWER: ") != 1:
                continue
            n_correct += 1
            key = " ".join(txt.split())
            if key in seen_txt:
                continue
            seen_txt.add(key)
            cands.append(txt)
        per_problem_correct.append(len(cands))
        cands.sort(key=len)               # prefer the shorter correct derivations
        for txt in cands[: args.max_keep_per_problem]:
            solved[p["src"]] += 1
            kept.append({"prompt": prompt,
                         "completion": txt.rstrip() + EOT,
                         "src": "rft_" + p["src"]})

    rng.shuffle(kept)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {"model": args.model, "n_problems": len(probs), "k": args.k,
             "n_generations": n_gen, "n_correct": n_correct,
             "pass_rate": n_correct / max(n_gen, 1),
             "problems_with_at_least_one_correct": sum(1 for c in per_problem_correct if c),
             "coverage": sum(1 for c in per_problem_correct if c) / len(probs),
             "n_kept": len(kept), "kept_by_source": dict(solved)}
    json.dump(stats, open(args.stats, "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
