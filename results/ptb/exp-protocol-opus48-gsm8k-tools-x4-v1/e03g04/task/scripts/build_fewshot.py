#!/usr/bin/env python3
"""Few-shot-augmented training set. To make stopping robust to the grader's
10-shot prompt (whose examples are joined by '\\n\\n' with NO stop token, priming
the model to continue), we prepend k random GSM8K-train few-shot examples --
formatted EXACTLY like the grader's sample_to_fewshot -- as a system message to a
fraction of the targets. The assistant target is unchanged (single ANSWER: N +
<end_of_turn>). This teaches: answer the FINAL question, then stop, regardless of
preceding Q/A blocks.

Few-shot pool uses GSM8K TRAIN (never test). Targets come from the RFT union.
"""
import argparse
import json
import random
import re

from datasets import load_dataset
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def fewshot_str(q, answer_raw):
    # replicate inspect_evals sample_to_fewshot: reasoning keeps <<>>; ANSWER: target
    parts = answer_raw.split("####")
    reasoning = "####".join(parts[:-1]).strip()
    target = parts[-1].strip().replace(",", "")
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="data/rft_sft.jsonl")
    ap.add_argument("--out", default="data/fs_sft.jsonl")
    ap.add_argument("--text-out", default="data/fs_text.jsonl")
    ap.add_argument("--n-human", type=int, default=7473)
    ap.add_argument("--n-rft", type=int, default=8500)
    ap.add_argument("--p-fewshot", type=float, default=0.5)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(SNAP)
    tmpl = open("templates/gemma3.jinja").read()

    # few-shot pool: raw GSM8K train (question, answer with reasoning+####)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    pool = [fewshot_str(r["question"].strip(), r["answer"]) for r in gsm]
    pool_toklen = [len(tok(s, add_special_tokens=False).input_ids) for s in pool]

    # targets: split human vs rft, subsample
    human, rft = [], []
    for l in open(args.targets):
        d = json.loads(l)
        (human if d.get("src") != "rft" else rft).append(d)
    random.shuffle(human); random.shuffle(rft)
    targets = human[: args.n_human] + rft[: args.n_rft]
    random.shuffle(targets)

    n_fs = 0
    truncated = 0
    with open(args.out, "w") as f, open(args.text_out, "w") as ft:
        for d in targets:
            row = {"prompt": d["prompt"], "completion": d["completion"], "answer": d["answer"]}
            comp_len = len(tok(d["completion"], add_special_tokens=False).input_ids)
            prompt_len = len(tok(d["prompt"], add_special_tokens=False).input_ids)
            budget = args.max_seq_len - comp_len - prompt_len - 32
            sys_text = ""
            if random.random() < args.p_fewshot and budget > 200:
                k = random.randint(args.kmin, args.kmax)
                idxs = random.sample(range(len(pool)), k)
                # greedily add shots that fit the token budget
                chosen, used = [], 0
                for j in idxs:
                    if used + pool_toklen[j] + 2 > budget:
                        continue
                    chosen.append(pool[j]); used += pool_toklen[j] + 2
                if chosen:
                    sys_text = "\n\n".join(chosen)
                    row["system"] = sys_text
                    n_fs += 1
            f.write(json.dumps(row) + "\n")
            doc = (sys_text + "\n" if sys_text else "") + d["prompt"] + "\n" + d["completion"]
            ft.write(json.dumps({"text": doc}) + "\n")

    print(f"total targets: {len(targets)} | with few-shot prefix: {n_fs} ({n_fs/len(targets):.0%})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
