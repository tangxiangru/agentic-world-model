#!/usr/bin/env python3
"""Rejection sampling: sample k solutions per GSM8K-train question, keep the correct ones."""
import argparse, json, os, re, random, collections
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)")


def extract(text):
    ms = ANS_RE.findall(text)
    if not ms:
        return None
    v = ms[-1].replace(",", "")
    try:
        f = float(v)
    except ValueError:
        return None
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.limit:
        ds = ds.select(range(args.limit))

    prompts, golds, questions = [], [], []
    for r in ds:
        q = r["question"].strip()
        gold = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        p = ("<bos><start_of_turn>user\n" + PROMPT_TEMPLATE.format(prompt=q)
             + "<end_of_turn>\n<start_of_turn>model\n")
        prompts.append(p)
        golds.append(gold)
        questions.append(q)

    llm = LLM(model=args.model, gpu_memory_utilization=0.88, max_model_len=2048,
              dtype="bfloat16", enable_prefix_caching=True, generation_config="vllm")
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, seed=1234,
                        stop_token_ids=[1, 106],
                        stop=["<end_of_turn>", "<start_of_turn>"])
    outs = llm.generate(prompts, sp)

    n_solved = 0
    kept = 0
    with open(args.out, "w") as f:
        for q, gold, o in zip(questions, golds, outs):
            texts = [c.text.strip() for c in o.outputs]
            good = []
            seen = set()
            for t in texts:
                v = extract(t)
                if v is None or abs(v - gold) > 1e-6:
                    continue
                # normalize: drop everything after the ANSWER line
                idx = t.rfind("ANSWER:")
                body = t[:idx].strip()
                nl = t[idx:].split("\n")[0].strip()
                t2 = body + "\n\n" + nl
                key = re.sub(r"\s+", " ", body)[:400]
                if key in seen:
                    continue
                seen.add(key)
                good.append(t2)
            if good:
                n_solved += 1
            for t in good:
                f.write(json.dumps({"question": q, "completion": t,
                                    "gold": gold, "n_correct": len(texts)}) + "\n")
                kept += 1
    print(f"solved {n_solved}/{len(questions)} = {n_solved/len(questions):.3f}; kept {kept} samples")


if __name__ == "__main__":
    main()
