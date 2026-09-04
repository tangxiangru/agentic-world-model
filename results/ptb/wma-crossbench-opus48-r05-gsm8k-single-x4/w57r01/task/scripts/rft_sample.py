#!/usr/bin/env python3
"""Rejection-sampling (STaR/RFT) generation: sample k CoT solutions per GSM8K
TRAIN question from a trained checkpoint via vLLM, keep those whose final
ANSWER matches gold, dedup, cap per question. Output prompt/completion JSONL in
the training format (completion ends in <end_of_turn>).
"""
import argparse, json, re, os
MATH_PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your '
    'response should be of the form "ANSWER: $ANSWER" (without quotes) where '
    '$ANSWER is the answer to the problem.\n\n{prompt}\n\nRemember to put your '
    'answer on its own line at the end in the form "ANSWER: $ANSWER" (without '
    'quotes) where $ANSWER is the answer to the problem, and you do not need to '
    'use a \\boxed command.\n\nReasoning:'
)

def gold_answer(ans):
    return ans.split("####")[-1].strip().replace(",", "")

def last_number(text):
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None

def norm_num(s):
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    chat_template = open(args.template).read()

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    prompts, metas = [], []
    for ex in gsm:
        user = MATH_PROMPT_TEMPLATE.format(prompt=ex["question"].strip())
        p = tok.apply_chat_template([{"role": "user", "content": user}],
                                    tokenize=False, add_generation_prompt=True,
                                    chat_template=chat_template)
        prompts.append(p)
        metas.append((p, norm_num(gold_answer(ex["answer"]))))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1536, dtype="bfloat16")
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95,
                        max_tokens=args.max_tokens, stop=["<end_of_turn>"])
    outs = llm.generate(prompts, sp)

    kept = 0
    with open(args.out, "w") as f:
        for (prompt, gold), out in zip(metas, outs):
            seen = set()
            n_this = 0
            for o in out.outputs:
                if n_this >= args.cap:
                    break
                text = o.text.strip()
                pred = last_number(text)
                if pred is None or norm_num(pred) != gold:
                    continue
                key = re.sub(r"\s+", " ", text)
                if key in seen:
                    continue
                seen.add(key)
                completion = f"{text}<end_of_turn>\n"
                f.write(json.dumps({"prompt": prompt, "completion": completion,
                                    "target": gold}) + "\n")
                kept += 1
                n_this += 1
    print(f"kept {kept} correct self-samples -> {args.out}")

if __name__ == "__main__":
    main()
