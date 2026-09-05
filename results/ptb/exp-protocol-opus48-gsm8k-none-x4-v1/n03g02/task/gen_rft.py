#!/usr/bin/env python3
"""Rejection sampling: generate solutions for GSM8K TRAIN with a fine-tuned model,
keep only those whose final ANSWER matches the gold answer. Output prompt/completion JSONL."""
import os, re, json, argparse
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from datasets import load_dataset
from vllm import LLM, SamplingParams

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-per-q", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=640)
    return p.parse_args()

def extract_answer(text):
    # take the FIRST 'ANSWER:' occurrence (model may ramble afterwards)
    m = re.search(r"ANSWER:\s*([^\n]+)", text)
    if not m: return None
    a = m.group(1).strip().replace(",", "").replace("$", "").strip().rstrip(".")
    return a

def norm(x):
    if x is None: return None
    x = x.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(x)
        if f == int(f): return str(int(f))
        return str(f)
    except:
        return x

def main():
    args = parse()
    ds = load_dataset("openai/gsm8k", "main")["train"]
    tok_path = args.model
    llm = LLM(model=args.model, tokenizer=tok_path, gpu_memory_utilization=0.85,
              max_model_len=2048, dtype="bfloat16", enforce_eager=False)
    tokenizer = llm.get_tokenizer()

    # build zero-shot prompts using the chat template
    prompts = []
    golds = []
    questions = []
    for ex in ds:
        gold = norm(ex["answer"].split("####")[1])
        user = MATH_PROMPT_TEMPLATE.format(prompt=ex["question"].strip())
        msgs = [{"role": "user", "content": user}]
        ptext = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(ptext)
        golds.append(gold)
        questions.append(ex["question"].strip())

    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, stop=None)
    print(f"Generating for {len(prompts)} questions, k={args.k}, temp={args.temp}")
    outs = llm.generate(prompts, sp)

    kept = 0
    n_solved = 0
    with open(args.out, "w") as f:
        for i, o in enumerate(outs):
            gold = golds[i]
            good = []
            seen = set()
            for comp in o.outputs:
                txt = comp.text
                # cut off any rambling after first answer line
                ans = extract_answer(txt)
                if ans is None: continue
                if norm(ans) != gold: continue
                # truncate solution at the first ANSWER line (keep clean single solution)
                mi = re.search(r"ANSWER:\s*[^\n]+", txt)
                sol = txt[:mi.end()].strip()
                # dedupe near-identical
                key = re.sub(r"\s+", " ", sol)[:120]
                if key in seen: continue
                seen.add(key)
                # normalize final answer line to gold
                sol = re.sub(r"ANSWER:\s*[^\n]+\s*$", f"ANSWER: {gold}", sol)
                good.append(sol)
            if good:
                n_solved += 1
            for sol in good[:args.max_per_q]:
                user = MATH_PROMPT_TEMPLATE.format(prompt=questions[i])
                rec = {"prompt": [{"role": "user", "content": user}],
                       "completion": [{"role": "assistant", "content": sol}]}
                f.write(json.dumps(rec) + "\n")
                kept += 1
    print(f"Solved (>=1 correct): {n_solved}/{len(prompts)} = {n_solved/len(prompts):.3f}")
    print(f"Kept solutions: {kept}")

if __name__ == "__main__":
    main()
