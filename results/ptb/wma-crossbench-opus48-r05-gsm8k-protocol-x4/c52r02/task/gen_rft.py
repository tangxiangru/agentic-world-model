#!/usr/bin/env python3
"""Rejection-sampling data generation for exp-04 (RFT/STaR).

Sample the exp-03 checkpoint on GSM8K *train* questions (NOT test), keep only
completions whose extracted answer matches the gold train answer. These are
verified-correct, on-distribution (<<>>-style) reasoning traces that we add to
the SFT set. Uses the SAME prompt template as training/eval (templates/gemma3.jinja
+ MATH_PROMPT_TEMPLATE). Answer extraction mimics the grader: last number in text.
"""
import argparse, json, re
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def last_num(s):
    ms = NUM.findall(s)
    if not ms:
        return None
    return ms[-1].replace(",", "").rstrip(".")

def gold_ans(answer):
    tail = answer.split("####")[-1].strip()
    return tail.replace(",", "").replace("$", "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpts/exp03/final")
    ap.add_argument("--out", default="data/rft_correct.jsonl")
    ap.add_argument("--n", type=int, default=4)          # samples per question
    ap.add_argument("--keep", type=int, default=2)       # max correct kept per question
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()
    ds = load_dataset("openai/gsm8k", "main", split="train")

    prompts, meta = [], []
    for r in ds:
        ptxt = MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip())
        rendered = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                           tokenize=False, add_generation_prompt=True)
        prompts.append(rendered)
        meta.append({"prompt": ptxt, "gold": gold_ans(r["answer"])})

    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106])
    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=1024,
              dtype="bfloat16")
    outs = llm.generate(prompts, sp)

    n_q_solved = 0
    kept = 0
    with open(args.out, "w") as f:
        for o, m in zip(outs, meta):
            gold = m["gold"]
            good = []
            for c in o.outputs:
                text = c.text.strip()
                # strict: exactly one clean answer, no runaway / restarted problem
                if text.count("ANSWER:") != 1:
                    continue
                if "Solve the following math problem" in text:
                    continue
                pred = last_num(text.split("ANSWER:")[-1])
                if pred is None:
                    continue
                if pred == gold:
                    good.append(text)
            if good:
                n_q_solved += 1
                # prefer shorter (cleaner) traces; keep up to args.keep unique
                seen = set()
                for text in sorted(good, key=len):
                    if text in seen:
                        continue
                    seen.add(text)
                    # ensure the completion ends right after the ANSWER line + stop token
                    comp = text
                    if not comp.endswith("<end_of_turn>"):
                        comp = comp + "<end_of_turn>"
                    f.write(json.dumps({"prompt": m["prompt"], "completion": comp,
                                        "target": gold}) + "\n")
                    kept += 1
                    if len([s for s in seen]) >= args.keep:
                        break
    print(f"questions with >=1 correct sample: {n_q_solved}/{len(meta)}")
    print(f"kept {kept} correct traces -> {args.out}")

if __name__ == "__main__":
    main()
