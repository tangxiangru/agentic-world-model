#!/usr/bin/env python3
"""Rejection-sampling: sample solutions for GSM8K *train* problems from a fine-tuned
checkpoint, keep the ones that reach the gold answer, dedupe by reasoning skeleton."""
import argparse, json, os, random, re
from collections import defaultdict
from datasets import load_dataset

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
EQ = re.compile(r"(-?[\d,]+\.?\d*)\s*([+\-*/x×÷])\s*(-?[\d,]+\.?\d*)\s*=\s*(-?[\d,]+\.?\d*)")


def last_number(s: str):
    m = NUM.findall(s.strip())
    if not m:
        return None
    v = m[-1].replace(",", "").rstrip(".")
    try:
        f = float(v)
    except ValueError:
        return None
    return f


def skeleton(s: str):
    return tuple(sorted(set(f"{a.replace(',','')}{op}{b.replace(',','')}={c.replace(',','')}"
                            for a, op, b, c in EQ.findall(s))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="work/rft.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-keep", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    problems = []
    for r in ds:
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        try:
            g = float(gold)
        except ValueError:
            continue
        problems.append((r["question"], g))
    print("problems", len(problems))

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
        tokenize=False, add_generation_prompt=True) for q, _ in problems]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=2048, dtype="bfloat16", enable_prefix_caching=True)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, seed=1234)
    outs = llm.generate(prompts, sp)

    kept = defaultdict(list)
    n_correct = 0
    n_total = 0
    solved = 0
    for (q, gold), o in zip(problems, outs):
        seen = set()
        any_ok = False
        cands = []
        for c in o.outputs:
            n_total += 1
            txt = c.text.strip()
            if c.finish_reason != "stop":
                continue
            if "ANSWER:" not in txt:
                continue
            v = last_number(txt)
            if v is None or abs(v - gold) > 1e-4:
                continue
            n_correct += 1
            any_ok = True
            sk = skeleton(txt)
            if sk in seen:
                continue
            seen.add(sk)
            cands.append(txt)
        if any_ok:
            solved += 1
        random.Random(hash(q) % 10**6).shuffle(cands)
        kept[q] = cands[:args.max_keep]

    print(f"pass rate {n_correct}/{n_total} = {n_correct/max(n_total,1):.3f}; solved {solved}/{len(problems)}")

    # fewshot pool for prompt-format diversity
    fewshot_pool = []
    for r in ds:
        parts = r["answer"].split("####")
        fewshot_pool.append((r["question"], "####".join(parts[:-1]).strip(), parts[-1].strip()))

    rng = random.Random(0)
    recs = []
    for q, sols in kept.items():
        for s in sols:
            recs.append((q, s))
    rng.shuffle(recs)
    with open(args.out, "w") as fh:
        for q, s in recs:
            msgs = []
            if rng.random() < args.fewshot_frac:
                k = rng.choices([2, 4, 10], weights=[0.4, 0.3, 0.3])[0]
                shots = rng.sample(fewshot_pool, k)
                msgs.append({"role": "system", "content": "\n\n".join(
                    f"{a}\n\nReasoning:\n{b}\n\nANSWER: {c}" for a, b, c in shots)})
            msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)})
            msgs.append({"role": "assistant", "content": s})
            fh.write(json.dumps({"messages": msgs, "source": "rft"}) + "\n")
    print("wrote", args.out, len(recs))
    with open(args.out + ".decon", "w") as fh:
        for q, s in recs:
            fh.write(json.dumps({"text": q + "\n" + s}) + "\n")


if __name__ == "__main__":
    main()
