#!/usr/bin/env python3
"""Checks that only a human can do, run before the GPU is booked.

1. template fidelity - render the graded conversation with the grader's own
   templates/gemma3.jinja and assert it is byte-identical to what build_data.py
   writes into the `prompt` field (pitfall: template_unreachable).
2. tokenisation - confirm the completion's final tokens are ANSWER, the number,
   and <end_of_turn>, and that <end_of_turn> is a single id in the eos set the
   grader's vLLM will stop on (pitfall: eos_mismatch).
3. length - p50/p99/max of prompt+completion tokens, and the share of rows that
   would truncate at --max-seq-len (pitfall: seq_len_truncation).
"""
import argparse
import hashlib
import json
import sys

from transformers import AutoTokenizer

TASK = "/home/ben/task"
SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{TASK}/data/sft_v1.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    tpl_path = f"{TASK}/templates/gemma3.jinja"
    tpl = open(tpl_path).read()
    print(f"grader template {tpl_path} sha256={hashlib.sha256(tpl.encode()).hexdigest()[:16]}")

    tok = AutoTokenizer.from_pretrained(SNAP)
    rows = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            rows.append(json.loads(line))

    ok = True

    # --- 1. template fidelity ------------------------------------------------
    r = rows[0]
    user = r["prompt"].split("<start_of_turn>user\n", 1)[1].rsplit("<end_of_turn>", 1)[0]
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        chat_template=tpl, tokenize=False, add_generation_prompt=True)
    if rendered != r["prompt"]:
        ok = False
        print("FAIL template fidelity")
        print("  grader :", repr(rendered[:120]), "...", repr(rendered[-60:]))
        print("  ours   :", repr(r["prompt"][:120]), "...", repr(r["prompt"][-60:]))
    else:
        print(f"PASS template fidelity - grader render == our prompt ({len(rendered)} chars)")

    # --- 2. stop token / answer marker --------------------------------------
    eot_ids = tok("<end_of_turn>", add_special_tokens=False)["input_ids"]
    print(f"     <end_of_turn> -> {eot_ids} (generation_config eos_token_id = [1, 106])")
    if eot_ids != [106]:
        ok = False
        print("FAIL <end_of_turn> is not the single id 106 the grader stops on")

    bad_stop = sum(1 for r in rows if not r["completion"].endswith("<end_of_turn>"))
    bad_mark = sum(1 for r in rows if r["completion"].count("ANSWER: ") != 1)
    print(f"{'PASS' if not bad_stop else 'FAIL'} stop token - {bad_stop}/{len(rows)} completions do not end with <end_of_turn>")
    print(f"{'PASS' if not bad_mark else 'FAIL'} answer marker - {bad_mark}/{len(rows)} completions do not contain 'ANSWER: ' exactly once")
    ok = ok and not bad_stop and not bad_mark

    # the grader takes the LAST whitespace-separated number in the completion
    import re
    bad_last = 0
    for r in rows:
        body = r["completion"][: -len("<end_of_turn>")]
        tail = body.rsplit("ANSWER: ", 1)[1]
        nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
        if len(nums) != 1 or nums[0] != tail.strip():
            bad_last += 1
    print(f"{'PASS' if not bad_last else 'FAIL'} last-number - {bad_last}/{len(rows)} completions have something other than the answer after 'ANSWER: '")
    ok = ok and not bad_last

    # --- 3. lengths ----------------------------------------------------------
    lens = []
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(r["completion"], add_special_tokens=False)["input_ids"]
        lens.append(len(p) + len(c))
    lens.sort()
    n = len(lens)
    trunc = sum(1 for x in lens if x > args.max_seq_len) / n
    print(f"     tokens p50={lens[n//2]} p90={lens[int(.9*n)]} p99={lens[int(.99*n)]} max={lens[-1]}")
    print(f"{'PASS' if trunc <= 0.02 else 'FAIL'} truncation - {trunc:.3%} of rows exceed max_seq_len={args.max_seq_len} (budget 2%)")
    ok = ok and trunc <= 0.02

    print("\n--- example prompt tail + completion ---")
    print(repr(rows[0]["prompt"][-200:]))
    print(repr(rows[0]["completion"][:300]))
    print("...", repr(rows[0]["completion"][-80:]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
