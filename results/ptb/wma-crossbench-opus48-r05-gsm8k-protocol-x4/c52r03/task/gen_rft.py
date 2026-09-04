#!/usr/bin/env python3
"""Rejection-sampling (STaR) generation for GSM8K TRAIN questions.
Samples k solutions per train question from a trained model via vLLM, keeps those
whose final answer matches the gold, dedups, and writes SFT jsonl {prompt,completion,gold,text}.
Uses ONLY GSM8K TRAIN questions + gold (never test).
"""
import argparse, json, re, os

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def last_number(s):
    nums = NUM.findall(s)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")

def norm(x):
    try:
        return float(str(x).replace(",", "").replace("$", "").rstrip("."))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="templates/gemma3.jinja")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=640)
    ap.add_argument("--max_keep_per_q", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    tmpl = open(args.template).read()
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts, golds, questions = [], [], []
    for r in ds:
        q = r["question"].strip()
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        user = {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}
        ps = tok.apply_chat_template([user], chat_template=tmpl, tokenize=False,
                                     add_generation_prompt=True)
        prompts.append(ps); golds.append(gold); questions.append(q)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, dtype="bfloat16", enforce_eager=False)
    # gemma3 assistant turns end with <end_of_turn> (token 106); it detokenizes to
    # empty text under skip_special_tokens, so a stop STRING never matches. Stop on
    # the token id (and also the string as a belt-and-suspenders fallback).
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop=["<end_of_turn>"],
                        stop_token_ids=[eot_id])
    outs = llm.generate(prompts, sp)

    kept = 0; nq = 0
    with open(args.out, "w") as f:
        for i, out in enumerate(outs):
            gold_f = norm(golds[i])
            seen = set(); nkept = 0
            for comp in out.outputs:
                text = comp.text.strip()
                pred = last_number(text)
                if pred is None or gold_f is None:
                    continue
                if norm(pred) is None or abs(norm(pred) - gold_f) > 1e-4:
                    continue
                # normalize final line to canonical ANSWER: gold
                body = re.split(r"\n?ANSWER:", text)[0].strip()
                if not body:
                    continue
                completion = f"{body}\nANSWER: {golds[i]}"
                key = body[:120]
                if key in seen:
                    continue
                seen.add(key)
                prompt = MATH_PROMPT_TEMPLATE.format(prompt=questions[i])
                f.write(json.dumps({"prompt": prompt, "completion": completion,
                                    "gold": golds[i], "text": prompt + "\n" + completion}) + "\n")
                nkept += 1; kept += 1
                if nkept >= args.max_keep_per_q:
                    break
            if nkept > 0:
                nq += 1
    print(f"[gen_rft] questions_with_>=1_correct={nq}/{len(outs)} kept_solutions={kept} -> {args.out}")

if __name__ == "__main__":
    main()
