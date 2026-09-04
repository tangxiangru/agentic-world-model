#!/usr/bin/env python3
"""Rejection-sampling generation (STaR/RFT).

Sample K solutions per GSM8K-TRAIN question from an SFT checkpoint, keep those
whose extracted final answer matches gold (same last-number rule as the grader),
dedup, cap per question. Output has the SAME schema as prepare_data.py so the
existing trainer/preflight consume it unchanged.
"""
import argparse, json, re
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def last_number(text):
    # mimic inspect match(numeric=True, location="end"): last number-looking token
    toks = re.split(r"\s+", text.strip())
    for w in reversed(toks):
        w2 = w.replace(",", "").replace("$", "").rstrip(".")
        if re.fullmatch(r"-?\d+(\.\d+)?", w2):
            return w2
    return None


def norm(x):
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--cap_per_q", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open("templates/gemma3.jinja").read()
    ds = load_dataset("openai/gsm8k", "main", split="train")

    prompts_text, targets, prompt_ids = [], [], []
    for r in ds:
        q = r["question"].strip()
        gold = norm(r["answer"].split("####")[1].strip().replace(",", ""))
        pt = MATH_PROMPT_TEMPLATE.format(prompt=q)
        pid = tok.apply_chat_template([{"role": "user", "content": pt}],
                                      chat_template=template,
                                      add_generation_prompt=True, tokenize=True)
        prompts_text.append(pt)
        targets.append(gold)
        prompt_ids.append(pid)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=1280, dtype="bfloat16", enforce_eager=False)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop_token_ids=[1, 106])
    token_prompts = [{"prompt_token_ids": pid} for pid in prompt_ids]
    outs = llm.generate(token_prompts, sampling_params=sp)

    kept, n_correct_q = [], 0
    for i, out in enumerate(outs):
        gold = targets[i]
        seen, good = set(), 0
        for o in out.outputs:
            text = o.text.strip()
            pred = norm(last_number(text))
            if pred is None or gold is None or pred != gold:
                continue
            # normalize whitespace key for dedup
            key = re.sub(r"\s+", " ", text)
            if key in seen:
                continue
            seen.add(key)
            comp = text
            if not comp.endswith("<end_of_turn>"):
                comp = comp + "<end_of_turn>"
            kept.append({"prompt": prompts_text[i], "completion": comp, "target": gold})
            good += 1
            if good >= args.cap_per_q:
                break
        if good > 0:
            n_correct_q += 1

    with open(args.out, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"[rft] questions={len(outs)} covered={n_correct_q} "
          f"({100*n_correct_q/len(outs):.1f}%) kept_samples={len(kept)}")


if __name__ == "__main__":
    main()
