#!/usr/bin/env python3
"""STaR rejection sampling with plain vLLM (time-boxed, safe BOS/stops).
Prompt token_ids come from apply_chat_template(tokenize=True) which emits a single
<bos> (CPU-verified); stop is <end_of_turn> id 106. Only GSM8K TRAIN is used."""
import os, re, json, random, argparse, gc
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()
CALC = re.compile(r"<<[^>]*>>")

def clean(ans):
    parts = ans.split("####")
    return CALC.sub("", parts[0]).strip(), parts[-1].replace(",", "").strip()

def fewshot_block(q, r, a): return f"{q}\n\nReasoning:\n{r}\n\nANSWER: {a}"

def norm_num(s):
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s); return str(int(f)) if f == int(f) else str(f)
    except Exception: return None

def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return norm_num(nums[-1]) if nums else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/rft_correct.jsonl")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--k-shot", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--keep-per-q", type=int, default=2)
    ap.add_argument("--template", default="templates/gemma3.jinja")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ct = open(args.template).read()
    tok = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    items = []
    for ex in ds:
        r, a = clean(ex["answer"])
        if re.fullmatch(r"-?\d+(\.\d+)?", a):
            items.append((ex["question"].strip(), r, a))
    n = len(items)

    all_ids, sidecar = [], []
    for i, (q, r, a) in enumerate(items):
        shots = "\n\n".join(fewshot_block(*items[j])
                            for j in rng.sample([x for x in range(n) if x != i], args.k_shot))
        msgs = [{"role": "system", "content": shots},
                {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      chat_template=ct)
        assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id, "double bos!"
        all_ids.append(ids)
        sidecar.append({"gold": a, "system": shots,
                        "prompt": MATH_PROMPT_TEMPLATE.format(prompt=q)})
    print(f"[sample] {n} prompts k_shot={args.k_shot} n={args.n} maxtok={args.max_tokens}", flush=True)

    params = SamplingParams(n=args.n, temperature=args.temp, top_p=0.95,
                            max_tokens=args.max_tokens, stop_token_ids=[106], seed=args.seed)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048, enforce_eager=True)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in all_ids], params)

    kept = 0; solved_q = 0; total_correct = 0
    with open(args.out, "w") as fout:
        for oi, out in enumerate(outputs):
            gold = sidecar[oi]["gold"]; seen = set(); q_kept = 0; had = False
            for comp in out.outputs:
                text = comp.text.strip()
                if text.count("ANSWER:") != 1: continue
                if last_number(text) != norm_num(gold): continue
                had = True; total_correct += 1
                key = re.sub(r"\s+", " ", text)
                if key in seen: continue
                seen.add(key)
                if q_kept >= args.keep_per_q: continue
                fout.write(json.dumps({"system": sidecar[oi]["system"],
                                       "prompt": sidecar[oi]["prompt"],
                                       "completion": text, "answer": gold}) + "\n")
                q_kept += 1; kept += 1
            if had: solved_q += 1
    print(f"[sample] solved {solved_q}/{n} ({100*solved_q/n:.1f}%); correct_draws {total_correct}; kept {kept}", flush=True)

    # explicit cleanup to avoid engine hang
    del llm; gc.collect()
    try: torch.cuda.empty_cache()
    except Exception: pass
    print("[sample] done, engine released", flush=True)

if __name__ == "__main__":
    main()
