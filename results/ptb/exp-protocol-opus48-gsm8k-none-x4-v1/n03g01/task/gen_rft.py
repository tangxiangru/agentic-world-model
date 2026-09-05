#!/usr/bin/env python3
"""Rejection sampling: generate solutions with an SFT model, keep correct ones."""
import os, json, re, argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def gold_answer(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def clean_completion(text):
    """Truncate at the FIRST 'ANSWER:' line; return (reasoning, pred) or (None,None)."""
    m = re.search(r"ANSWER:\s*([^\n]*)", text)
    if not m:
        return None, None
    reasoning = text[:m.start()].rstrip()
    seg = m.group(1)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", seg)
    if not nums:
        return reasoning, None
    return reasoning, nums[-1].replace(",", "").rstrip(".")


def num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except Exception:
        return str(a) == str(b)


def fewshot_block(q, reasoning_raw, final):
    return f"{q}\n\nReasoning:\n{reasoning_raw}\n\nANSWER: {final}"


def render(prompt):
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="rft_data.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--maxtok", type=int, default=640)
    ap.add_argument("--keep", type=int, default=4, help="max correct kept per question")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kshot", type=int, default=0, help="num few-shot exemplars in prompt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    ds = load_dataset("openai/gsm8k", "main")["train"]
    # exemplar pool for few-shot prefixes (raw reasoning with <<>> to match eval)
    exemplars = []
    for r in ds:
        parts = r["answer"].split("####")
        exemplars.append((r["question"].strip(), "####".join(parts[:-1]).strip(),
                          parts[-1].strip().replace(",", "")))
    if args.limit:
        ds = ds.select(range(args.limit))
    questions = [r["question"].strip() for r in ds]
    golds = [gold_answer(r["answer"]) for r in ds]

    def build_prompt(q):
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        if args.kshot > 0:
            picks = []
            while len(picks) < args.kshot:
                e = exemplars[random.randrange(len(exemplars))]
                if e[0] != q:
                    picks.append(e)
            sysc = "\n\n".join(fewshot_block(*p) for p in picks)
            user = sysc + "\n\n" + user
        return render(user)

    prompts = [build_prompt(q) for q in questions]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048, enable_prefix_caching=True)
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=0.95, top_k=64,
                        max_tokens=args.maxtok, stop=["<end_of_turn>"])
    outs = llm.generate(prompts, sp)

    kept = 0
    n_solved = 0
    with open(args.out, "w") as f:
        for i, o in enumerate(outs):
            gold = golds[i]
            seen = set()
            good = []
            for comp in o.outputs:
                reasoning, pred = clean_completion(comp.text.strip())
                if reasoning is None or pred is None:
                    continue
                if not num_eq(pred, gold):
                    continue
                if len(reasoning) < 5:  # skip degenerate no-reasoning answers
                    continue
                key = re.sub(r"\s+", " ", reasoning)[:200]
                if key in seen:
                    continue
                seen.add(key)
                good.append(f"{reasoning}\n\nANSWER: {gold}")
                if len(good) >= args.keep:
                    break
            if good:
                n_solved += 1
            for g in good:
                f.write(json.dumps({"prompt": MATH_PROMPT_TEMPLATE.format(prompt=questions[i]),
                                    "completion": g, "question": questions[i],
                                    "answer": gold}) + "\n")
                kept += 1
    print(f"solved {n_solved}/{len(questions)} questions; kept {kept} solutions")


if __name__ == "__main__":
    main()
