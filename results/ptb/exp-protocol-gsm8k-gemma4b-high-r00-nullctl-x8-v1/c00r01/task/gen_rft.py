#!/usr/bin/env python3
"""Rejection-sampling data generation: sample solutions from the current model on
GSM8K-train (and OpenMathInstruct augmented-gsm8k) problems, keep the correct ones."""
from __future__ import annotations
import argparse, glob, json, os, random, re
import pandas as pd

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*([^\n]*)")


def norm_num(s):
    s = str(s).strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    s = s.strip()
    try:
        f = float(s)
    except (ValueError, OverflowError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
        return None
    return str(int(f)) if f == int(f) else str(round(f, 6))


def extract(text):
    ms = ANS_RE.findall(text)
    if not ms:
        return None
    return norm_num(ms[-1])


def build_prompt(q, prefix=""):
    return ("<bos><start_of_turn>user\n" + prefix
            + MATH_PROMPT_TEMPLATE.format(prompt=q)
            + "<end_of_turn>\n<start_of_turn>model\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k-aug", type=int, default=4)
    ap.add_argument("--n-aug", type=int, default=25000)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--keep-per-problem", type=int, default=4)
    ap.add_argument("--fewshot-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stats-out", default="runs/rft_stats.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main")["train"]
    problems = []  # (question, gold, k, tag)
    fewshot_pool = []
    for rec in gsm:
        reasoning, tgt = rec["answer"].split("####")
        g = norm_num(tgt)
        if g is None:
            continue
        problems.append((rec["question"].strip(), g, args.k, "gsm8k_train"))
        fewshot_pool.append((rec["question"].strip(),
                             reasoning.strip() + "\n\nANSWER: " + g))

    if args.n_aug > 0:
        files = sorted(glob.glob("/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                                 "snapshots/*/data/*.parquet"))
        seen = set()
        aug = []
        for f in files:
            df = pd.read_parquet(f, columns=["problem", "expected_answer", "problem_source"])
            df = df[df.problem_source == "augmented_gsm8k"]
            for prob, ans, _ in df.itertuples(index=False):
                if prob in seen or len(prob) > 1200:
                    continue
                a = norm_num(ans)
                if a is None:
                    continue
                seen.add(prob)
                aug.append((prob.strip(), a, args.k_aug, "aug_gsm8k"))
            if len(aug) > args.n_aug * 3:
                break
        rng.shuffle(aug)
        problems += aug[:args.n_aug]
    print("problems:", len(problems), flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=2048, enforce_eager=False, seed=args.seed,
              disable_log_stats=True)

    prompts, meta = [], []
    for q, g, k, tag in problems:
        prompts.append(build_prompt(q))
        meta.append((q, g, k, tag))
    sp_groups = {}
    for i, (q, g, k, tag) in enumerate(meta):
        sp_groups.setdefault(k, []).append(i)

    results = [None] * len(prompts)
    for k, idxs in sp_groups.items():
        sp = SamplingParams(n=k, temperature=args.temp, top_p=0.95, top_k=64,
                            max_tokens=args.max_tokens, stop=["<end_of_turn>"])
        outs = llm.generate([prompts[i] for i in idxs], sp)
        for i, o in zip(idxs, outs):
            results[i] = [c.text for c in o.outputs]

    kept, n_solved, n_total, per_tag = [], 0, 0, {}
    solve_rate = {}
    for (q, g, k, tag), cands in zip(meta, results):
        n_total += 1
        good, sigs = [], set()
        for c in cands:
            if extract(c) != g:
                continue
            body = c.strip()
            if len(body) < 10 or len(body) > 2200:
                continue
            sig = tuple(re.findall(r"-?\d+\.?\d*", body))
            if sig in sigs:
                continue
            sigs.add(sig)
            good.append(body)
        solve_rate[q[:120]] = (len([c for c in cands if extract(c) == g]), k)
        if good:
            n_solved += 1
            per_tag[tag] = per_tag.get(tag, 0) + 1
            good.sort(key=len)
            for b in good[:args.keep_per_problem]:
                kept.append((q, b, tag))
    print(f"solved {n_solved}/{n_total}; kept {len(kept)} samples; per_tag {per_tag}", flush=True)

    def make_prefix(k, exclude_q):
        picks = []
        while len(picks) < k:
            q, s = fewshot_pool[rng.randrange(len(fewshot_pool))]
            if q == exclude_q:
                continue
            picks.append(f"{q}\n\nReasoning:\n{s}")
        return "\n\n".join(picks) + "\n\n"

    rng.shuffle(kept)
    with open(args.out, "w") as fh:
        for q, body, tag in kept:
            prefix = make_prefix(rng.choice([1, 2, 3, 4, 5, 8, 10]), q) \
                if rng.random() < args.fewshot_frac else ""
            fh.write(json.dumps({
                "prompt": "<start_of_turn>user\n" + prefix
                          + MATH_PROMPT_TEMPLATE.format(prompt=q)
                          + "<end_of_turn>\n<start_of_turn>model\n",
                "completion": body + "<end_of_turn>\n",
                "source": "rft_" + tag}) + "\n")
    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
    json.dump({"solved": n_solved, "total": n_total, "kept": len(kept),
               "unsolved_questions": [q for q, (c, k) in solve_rate.items() if c == 0][:50]},
              open(args.stats_out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
