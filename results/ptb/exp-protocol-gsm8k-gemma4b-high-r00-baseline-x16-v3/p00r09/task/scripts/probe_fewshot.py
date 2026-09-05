#!/usr/bin/env python3
"""Diagnostic: does the grader's 10-shot system prefix help or hurt the tuned model?

Scores the same held-out GSM8K *train* questions greedily under three contexts:
  zero-shot, the exact 10-shot block the grader builds (gsm8k train, seed 42),
  and a 2-shot block. GSM8K test is never read.
"""
import argparse
import json
import re

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = datasets.load_dataset("openai/gsm8k", "main", split="train")
    # the grader's own few-shot selection: shuffle(seed=42), take 10
    shot_ds = train.shuffle(seed=42).select(range(10))
    shots = []
    for r in shot_ds:
        parts = r["answer"].split("####")
        shots.append((r["question"].strip(), "####".join(parts[:-1]).strip(),
                      parts[-1].strip()))
    shot_qs = {q for q, _, _ in shots}

    # held-out probe items: last 400 train items, none of them a few-shot example
    probe = []
    for r in list(train)[::-1]:
        q = r["question"].strip()
        if q in shot_qs:
            continue
        probe.append((q, r["answer"].split("####")[-1].strip().replace(",", "")))
        if len(probe) >= args.n:
            break

    def block(k):
        return "\n\n".join(f"{q}\n\nReasoning:\n{reas}\n\nANSWER: {a}"
                           for q, reas, a in shots[:k])

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open("/home/ben/task/templates/gemma3.jinja").read()
    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=4096, seed=0)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=1024, stop_token_ids=[1, 106])

    res = {}
    for name, k in (("zero_shot", 0), ("two_shot", 2), ("ten_shot", 10)):
        prompts = []
        for q, _ in probe:
            msgs = []
            if k:
                msgs.append({"role": "system", "content": block(k)})
            msgs.append({"role": "user",
                         "content": MATH_PROMPT_TEMPLATE.format(prompt=q)})
            text = tok.apply_chat_template(msgs, chat_template=template,
                                           tokenize=False, add_generation_prompt=True)
            prompts.append({"prompt_token_ids":
                            tok(text, add_special_tokens=False)["input_ids"]})
        outs = llm.generate(prompts, sp)
        ok = 0
        malformed = 0
        for (q, gold), o in zip(probe, outs):
            txt = o.outputs[0].text.strip()
            m = ANS_RE.search(txt)
            if not m:
                malformed += 1
                continue
            try:
                if abs(float(m.group(1).replace(",", "")) - float(gold)) < 1e-6:
                    ok += 1
            except ValueError:
                pass
        res[name] = {"acc": ok / len(probe), "n": len(probe), "malformed": malformed}
        print(name, res[name], flush=True)

    json.dump({"model": args.model, "results": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
