#!/usr/bin/env python3
"""Filter the raw SFT corpus and pre-tokenize it against the grader's own template.

Guards, in order:
  * completion contains "ANSWER: " exactly once and ends with it on the last line
  * no leftover \\boxed / #### (a second answer marker the grader could read first)
  * renders through templates/gemma3.jinja (byte-for-byte the grader's template)
  * total length <= --max-seq-len, so no row loses its completion to truncation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
STOP = "<end_of_turn>"
ANSWER_LINE = re.compile(r"\nANSWER: (-?\d[\d,]*(?:\.\d+)?)$")


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(SNAP)
    with open(TEMPLATE) as f:
        tmpl = f.read()
    tok.chat_template = tmpl
    return tok, hashlib.sha256(tmpl.encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/sft_v1.jsonl")
    ap.add_argument("--out-jsonl", default="data/sft_v1_clean.jsonl")
    ap.add_argument("--out-npz", default="data/sft_v1_tokens.npz")
    ap.add_argument("--max-seq-len", type=int, default=2560)
    args = ap.parse_args()

    tok, thash = load_tokenizer()
    print("template sha256[:12]", thash)

    kept, dropped = [], {"marker": 0, "boxed": 0, "len": 0, "mask": 0}
    ids_all, lens, plens = [], [], []
    with open(args.inp) as fh:
        for line in fh:
            r = json.loads(line)
            c = r["completion"]
            if c.count("ANSWER:") != 1 or not ANSWER_LINE.search(c):
                dropped["marker"] += 1
                continue
            if "boxed" in c or "####" in c:
                dropped["boxed"] += 1
                continue
            prompt = tok.apply_chat_template(
                r["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            full = prompt + c + STOP
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            if len(f_ids) > args.max_seq_len:
                dropped["len"] += 1
                continue
            if len(f_ids) - len(p_ids) < 8:
                dropped["mask"] += 1
                continue
            ids_all.append(np.array(f_ids, dtype=np.int32))
            lens.append(len(f_ids))
            plens.append(len(p_ids))
            kept.append(r)

    with open(args.out_jsonl, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")

    flat = np.concatenate(ids_all)
    offs = np.zeros(len(ids_all) + 1, dtype=np.int64)
    offs[1:] = np.cumsum(lens)
    np.savez(args.out_npz, flat=flat, offsets=offs,
             prompt_lens=np.array(plens, dtype=np.int32))

    lens_arr = np.array(lens)
    print(f"kept {len(kept)} dropped {dropped}")
    print(f"total tokens {flat.size/1e6:.1f}M  len p50={np.median(lens_arr):.0f} "
          f"p95={np.percentile(lens_arr,95):.0f} max={lens_arr.max()}")
    print(f"prompt len p50={np.median(plens):.0f} max={max(plens)}")
    # last-token check: every row must end with the stop token
    stop_id = tok.convert_tokens_to_ids(STOP)
    bad = sum(1 for a in ids_all if a[-1] != stop_id)
    print(f"rows not ending in {STOP} (id {stop_id}): {bad}")


if __name__ == "__main__":
    main()
