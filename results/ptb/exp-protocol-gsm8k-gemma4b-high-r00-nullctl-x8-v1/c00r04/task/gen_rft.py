#!/usr/bin/env python3
"""Rejection-sampling fine-tuning data generation (STaR/RFT).

Samples solutions from a fine-tuned checkpoint on GSM8K *train* problems (and
OpenMathInstruct-2 augmented-gsm8k problems, which are themselves derived from
the GSM8K train split), keeps the ones whose final answer matches the reference,
and writes them out in the SFT record format.
"""
import argparse, json, os, random, re, sys
from collections import defaultdict

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def to_float(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except Exception:
        return None


def extract_answer(text):
    m = re.findall(r"ANSWER:\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return to_float(m[-1])
    m = NUM.findall(text)
    return to_float(m[-1]) if m else None


def fmt_answer(f):
    if abs(f - round(f)) < 1e-9:
        return f"{int(round(f)):,}"
    return ("%.6f" % f).rstrip("0").rstrip(".")


def norm_q(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft.jsonl")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-aug", type=int, default=20000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    test = json.load(open("/home/ben/test_data.json"))
    test_norm = {norm_q(x["question"]) for x in test}

    from datasets import load_dataset
    problems = []
    gs = load_dataset("openai/gsm8k", "main", split="train")
    for q, a in zip(gs["question"], gs["answer"]):
        if norm_q(q) in test_norm:
            continue
        gold = to_float(a.split("####")[-1].strip())
        if gold is not None:
            problems.append({"question": q.strip(), "gold": gold, "src": "rft_gsm8k"})

    if args.n_aug:
        import pyarrow.parquet as pq
        rows = pq.read_table("data/omi2_augmented_gsm8k.parquet").to_pylist()
        rng = random.Random(7)
        rng.shuffle(rows)
        seen = set()
        for r in rows:
            if len(seen) >= args.n_aug:
                break
            q = r["problem"].strip()
            k = norm_q(q)
            if k in seen or k in test_norm:
                continue
            gold = to_float(r["expected_answer"])
            if gold is None:
                continue
            seen.add(k)
            problems.append({"question": q, "gold": gold, "src": "rft_aug"})
    print("problems:", len(problems))

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=p["question"])}],
            tokenize=False, add_generation_prompt=True)
        for p in problems
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1600, dtype="bfloat16", max_num_seqs=512)
    sp = SamplingParams(n=args.n_samples, temperature=args.temp, top_p=0.95,
                        top_k=64, max_tokens=args.max_tokens, seed=0,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept, n_solved, per_prob_correct = [], 0, []
    for p, o in zip(problems, outs):
        good, seen_txt = [], set()
        for c in o.outputs:
            if c.finish_reason != "stop":
                continue
            t = c.text.strip()
            v = extract_answer(t)
            if v is None or abs(v - p["gold"]) > 1e-6:
                continue
            # normalise the final line so every kept target ends the same way
            body = t.split("ANSWER:")[0].strip()
            if len(body) < 20:
                continue
            sol = body + "\n\nANSWER: " + fmt_answer(p["gold"])
            key = re.sub(r"\s+", " ", body)[:400]
            if key in seen_txt:
                continue
            seen_txt.add(key)
            good.append(sol)
        per_prob_correct.append(len(good))
        if good:
            n_solved += 1
            good.sort(key=len)
            for sol in good[: args.keep_per_problem]:
                kept.append({"question": p["question"], "solution": sol,
                             "answer": fmt_answer(p["gold"]), "src": p["src"]})

    print(f"solved {n_solved}/{len(problems)} = {n_solved/len(problems):.3f}")
    print("kept records:", len(kept))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump({"n_problems": len(problems), "solved": n_solved,
                       "kept": len(kept)}, f)
    # unsolved problems (useful for a later, harder round)
    with open(args.out.replace(".jsonl", "_unsolved.jsonl"), "w") as f:
        for p, c in zip(problems, per_prob_correct):
            if c == 0:
                f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
