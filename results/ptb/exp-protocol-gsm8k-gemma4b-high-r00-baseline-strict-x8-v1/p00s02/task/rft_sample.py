"""Rejection-sampling fine-tuning data: sample the current model, keep what is right.

Problems come from the same GSM8K-TRAIN-derived pool as the SFT data (never the
benchmark test split). Solutions are the model's own, filtered by exact match of
the last numeric token against the known answer -- the same rule the grader uses.
"""
import argparse
import json
import os
import random
import re

from transformers import AutoTokenizer

from probe_eval import MATH_PROMPT_TEMPLATE, SNAP, TEMPLATE, last_number, norm_num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="/home/ben/task/data/sft_omi2_gsm8k_big_clean.jsonl")
    ap.add_argument("--n-problems", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    # one entry per distinct problem, with its known answer
    probs = {}
    for line in open(args.pool):
        d = json.loads(line)
        probs.setdefault(d["prompt"], d["answer"])
    items = sorted(probs.items())
    random.Random(args.seed).shuffle(items)
    items = items[: args.n_problems]
    print(f"[rft] {len(items)} distinct problems, k={args.k}", flush=True)

    tok = AutoTokenizer.from_pretrained(SNAP)
    template = open(TEMPLATE).read()
    prompts = [tok.apply_chat_template([{"role": "user", "content": u}],
                                       chat_template=template, tokenize=False,
                                       add_generation_prompt=True) for u, _ in items]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=1536,
              dtype="bfloat16", disable_log_stats=True)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, top_k=64, n=args.k,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106], seed=args.seed)
    outs = llm.generate(prompts, sp)

    n_kept = 0
    n_solved = 0
    pass_hist = [0] * (args.k + 1)
    rows = []
    for (user, gold), o in zip(items, outs):
        good = []
        for c in o.outputs:
            if c.finish_reason == "length":
                continue
            p = last_number(c.text)
            if p is not None and norm_num(p) == norm_num(gold) and c.text.count("ANSWER:") == 1:
                good.append(c.text.strip())
        pass_hist[len(good)] += 1
        n_solved += bool(good)
        # dedup, prefer the shortest correct chains (less rambling, fewer chances to slip)
        uniq = sorted(set(good), key=len)[: args.keep_per_problem]
        for t in uniq:
            rows.append({"prompt": user, "completion": t + "<end_of_turn>",
                         "answer": gold, "source": "synthetic:self"})
            n_kept += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"model": args.model, "n_problems": len(items), "k": args.k,
             "temperature": args.temperature, "kept_rows": n_kept,
             "problems_with_at_least_one_correct": n_solved,
             "solve_rate": n_solved / len(items),
             "pass_at_k_histogram": pass_hist}
    json.dump(stats, open(args.stats_out, "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
