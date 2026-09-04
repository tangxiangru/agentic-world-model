#!/usr/bin/env python3
"""Render Magicoder rows with the grader's gemma3.jinja template into a
prompt/completion JSONL that the trainer consumes verbatim. The `completion`
field ends exactly with <end_of_turn> so preflight's stop-token check verifies
the real training target (train == check == grader rendering)."""
import argparse, json, os
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--in", dest="inp", default="data/train_raw.jsonl")
    ap.add_argument("--out", default="data/sft_final.jsonl")
    ap.add_argument("--template", default="templates/gemma3.jinja")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.chat_template = open(a.template).read()

    rows = [json.loads(l) for l in open(a.inp)]
    n_written = 0
    with open(a.out, "w") as f:
        for r in rows:
            user = {"role": "user", "content": r["instruction"]}
            asst = {"role": "assistant", "content": r["response"]}
            prompt = tok.apply_chat_template([user], add_generation_prompt=True, tokenize=False)
            full = tok.apply_chat_template([user, asst], add_generation_prompt=False, tokenize=False)
            if not full.startswith(prompt):
                continue
            completion = full[len(prompt):].rstrip("\n")   # ends with <end_of_turn>
            if not completion.endswith("<end_of_turn>"):
                continue
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "text": r["text"], "src": r.get("src")}) + "\n")
            n_written += 1
    print(f"wrote {n_written} rows -> {a.out}")

if __name__ == "__main__":
    main()
