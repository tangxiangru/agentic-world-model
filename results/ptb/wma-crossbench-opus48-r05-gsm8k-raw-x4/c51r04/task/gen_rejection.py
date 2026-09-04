#!/usr/bin/env python3
"""Rejection sampling (STaR): sample solutions from an SFT model on GSM8K TRAIN
problems, keep only those whose final ANSWER matches the gold answer.
Outputs rej_data.jsonl with {"question","response","src"}.
"""
import re, json, argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def gold_num(ans):
    t = ans.split("####")[-1].strip().replace(",", "").replace("$", "").strip()
    return t

def norm(x):
    if x is None: return None
    x = x.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", x)
    if not m: return None
    try:
        v = float(m.group(0))
        return v
    except: return None

ANS_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

def extract_and_truncate(text):
    """Return (answer_str, clean_text) truncated right after the first
    'ANSWER: ...' line. None if no ANSWER line found."""
    m = ANS_RE.search(text)
    if not m:
        return None, None
    ans = m.group(1).strip()
    clean = text[: m.end()].rstrip()
    return ans, clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sft_run1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=640)
    ap.add_argument("--keep_per_q", type=int, default=4)
    ap.add_argument("--out", default="rej_data.jsonl")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)
    bos = tok.bos_token

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    if args.limit > 0:
        gsm = gsm.select(range(args.limit))
    questions = [r["question"].strip() for r in gsm]
    golds = [gold_num(r["answer"]) for r in gsm]

    prompts = []
    for q in questions:
        user = MATH_PROMPT_TEMPLATE.format(prompt=q)
        p = f"{bos}<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
        prompts.append(p)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=1536, enforce_eager=False)
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=0.95, top_k=64,
                        max_tokens=args.max_tokens, stop=["<end_of_turn>"])

    outputs = llm.generate(prompts, sp)

    kept = 0
    solved_q = 0
    with open(args.out, "w") as f:
        for i, out in enumerate(outputs):
            gold = norm(golds[i])
            seen = set()
            n_this = 0
            any_correct = False
            for comp in out.outputs:
                txt = comp.text.strip()
                a, clean = extract_and_truncate(txt)
                if a is None: continue
                if norm(a) is None: continue
                # reject degenerate / too-short reasoning
                if len(clean) < 25: continue
                if gold is not None and abs(norm(a) - gold) < 1e-6:
                    any_correct = True
                    # dedup by normalized whitespace of reasoning
                    key = re.sub(r"\s+", " ", clean)
                    if key in seen: continue
                    seen.add(key)
                    if n_this >= args.keep_per_q: continue
                    f.write(json.dumps({"question": questions[i], "response": clean,
                                        "src": "rej"}) + "\n")
                    n_this += 1
                    kept += 1
            if any_correct: solved_q += 1
    print(f"kept {kept} traces; solved {solved_q}/{len(questions)} problems "
          f"({100*solved_q/len(questions):.1f}% coverage)")

if __name__ == "__main__":
    main()
