#!/usr/bin/env python3
"""Build GSM8K-style SFT data. CPU-only. No model construction.

Outputs (data/):
  gsm8k_train.jsonl   - GSM8K train only, {prompt, completion}
  mix_train.jsonl     - GSM8K train + MetaMath GSM types, {prompt, completion}
  *_check.jsonl       - {question, answer} raw text, for contamination_check.py

Formatting: prompt = gemma3 chat render of MATH_PROMPT_TEMPLATE(question) with
add_generation_prompt; completion = cleaned reasoning + '\nANSWER: <N>' + '<end_of_turn>'.
Single answer marker; ends with the grader stop token.
"""
import os, re, json, argparse, random
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from datasets import load_dataset
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
STOP = "<end_of_turn>"

# Exactly the grader's per-question wrapper (inspect_evals/gsm8k.py)
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

_tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
_tok.chat_template = open(TEMPLATE_PATH).read()

def render_prompt(question: str) -> str:
    user = MATH_PROMPT_TEMPLATE.format(prompt=question.strip())
    return _tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )

def clean_num(s: str):
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.strip().rstrip(".")
    # keep a leading minus and digits/decimal
    m = re.match(r"^-?\d+(?:\.\d+)?$", s)
    return s if m else None

def gsm8k_completion(answer: str):
    sol = re.sub(r"<<[^>]*>>", "", answer)
    if "####" not in sol:
        return None
    reasoning, final = sol.rsplit("####", 1)
    final = clean_num(final)
    if final is None:
        return None
    reasoning = reasoning.strip()
    if not reasoning:
        return None
    return f"{reasoning}\nANSWER: {final}{STOP}"

def metamath_completion(response: str):
    # response ends with "#### N\nThe answer is: N" (GSM types)
    m = re.search(r"The answer is:\s*(.+?)\s*$", response.strip())
    if not m:
        return None
    final = clean_num(m.group(1))
    if final is None:
        return None
    body = response
    # cut trailing markers
    idx = body.find("\n####")
    if idx == -1:
        idx = body.find("####")
    if idx != -1:
        body = body[:idx]
    else:
        idx = body.rfind("The answer is:")
        body = body[:idx]
    reasoning = re.sub(r"<<[^>]*>>", "", body).strip()
    if not reasoning or len(reasoning) < 15:
        return None
    if "ANSWER:" in reasoning:
        return None
    return f"{reasoning}\nANSWER: {final}{STOP}"

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metamath-cap", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- GSM8K train ----
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm_rows, gsm_check = [], []
    for ex in gsm:
        comp = gsm8k_completion(ex["answer"])
        if comp is None:
            continue
        gsm_rows.append({"prompt": render_prompt(ex["question"]), "completion": comp})
        gsm_check.append({"question": ex["question"], "answer": ex["answer"]})
    print(f"GSM8K train usable: {len(gsm_rows)}")
    write_jsonl("data/gsm8k_train.jsonl", gsm_rows)
    write_jsonl("data/gsm8k_train_check.jsonl", gsm_check)

    # ---- MetaMath GSM types ----
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    gsm_types = {"GSM_Rephrased", "GSM_AnsAug", "GSM_FOBAR", "GSM_SV"}
    mm_rows, mm_check, seen = [], [], set()
    idxs = list(range(len(mm)))
    rng.shuffle(idxs)
    for i in idxs:
        if len(mm_rows) >= args.metamath_cap:
            break
        ex = mm[i]
        if ex["type"] not in gsm_types:
            continue
        q = ex["query"].strip()
        key = q.lower()
        if key in seen:
            continue
        comp = metamath_completion(ex["response"])
        if comp is None:
            continue
        seen.add(key)
        mm_rows.append({"prompt": render_prompt(q), "completion": comp})
        mm_check.append({"question": q, "answer": ex["response"]})
    print(f"MetaMath GSM usable (capped): {len(mm_rows)}")

    mix = gsm_rows + mm_rows
    rng.shuffle(mix)
    write_jsonl("data/mix_train.jsonl", mix)
    mix_check = gsm_check + mm_check
    write_jsonl("data/mix_train_check.jsonl", mix_check)
    print(f"MIX total: {len(mix)}")

    # length stats (chars) for the completion + prompt
    import numpy as np
    def stats(rows, name):
        toks = [len(_tok(r["prompt"] + r["completion"])["input_ids"]) for r in rows[:1500]]
        toks = np.array(toks)
        print(f"[{name}] token len (first {len(toks)}): p50={np.percentile(toks,50):.0f} "
              f"p95={np.percentile(toks,95):.0f} p99={np.percentile(toks,99):.0f} max={toks.max()}")
    stats(gsm_rows, "gsm8k")
    stats(mix, "mix")
    # show one example
    print("\n=== EXAMPLE prompt ===\n", gsm_rows[0]["prompt"])
    print("=== EXAMPLE completion ===\n", gsm_rows[0]["completion"])

if __name__ == "__main__":
    main()
