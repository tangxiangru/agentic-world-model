#!/usr/bin/env python3
"""Rejection-sampling data generation with vLLM from a fine-tuned checkpoint."""
import argparse, json, os, random, re

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_num(s):
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return round(v, 6)


def extract_answer(text):
    idx = text.rfind("ANSWER:")
    if idx < 0:
        return None
    tail = text[idx + len("ANSWER:"):].strip().split("\n")[0].strip()
    n = norm_num(tail)
    if n is not None:
        return n
    m = NUM_RE.findall(tail)
    return norm_num(m[-1]) if m else None


def truncate_after_answer(text):
    """Cut the completion right after the first 'ANSWER: ...' line."""
    idx = text.find("ANSWER:")
    if idx < 0:
        return None
    nl = text.find("\n", idx)
    return (text if nl < 0 else text[:nl]).rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, help="jsonl with question/answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(args.questions)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])}],
            tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    llm = LLM(model=args.model, gpu_memory_utilization=0.88, max_model_len=2048,
              enable_prefix_caching=True, seed=args.seed)
    # <end_of_turn> (106) is not picked up automatically by the offline API
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=None,
                        stop_token_ids=[1, 106])
    outs = llm.generate(prompts, sp)

    kept, solved, total = [], 0, 0
    hard = []
    for r, o in zip(rows, outs):
        gold = norm_num(str(r["answer"]))
        total += 1
        good, seen = [], set()
        for c in o.outputs:
            t = truncate_after_answer(c.text.strip())
            if t is None:
                continue
            a = extract_answer(t)
            if gold is not None and a is not None and abs(a - gold) < 1e-6:
                if t not in seen:
                    seen.add(t)
                    good.append(t)
        if good:
            solved += 1
            good.sort(key=len)
            for t in good[: args.keep_per_problem]:
                kept.append({"question": r["question"], "solution": t,
                             "answer": str(r["answer"]), "src": "rft"})
        else:
            hard.append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    hp = args.out.replace(".jsonl", "_hard.jsonl")
    with open(hp, "w") as f:
        for r in hard:
            f.write(json.dumps(r) + "\n")
    msg = {"total": total, "solved": solved, "pass_rate": solved / max(total, 1),
           "kept": len(kept), "hard": len(hard)}
    print(json.dumps(msg))
    if args.stats:
        json.dump(msg, open(args.stats, "w"))


if __name__ == "__main__":
    main()
