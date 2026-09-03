#!/usr/bin/env python3
"""Pre-launch proof for the two pitfalls a machine check cannot close here.

1. eos_mismatch  - the jsonl 'completion' field deliberately does NOT carry the
   terminator; prepare_tokens.py appends <end_of_turn> when it tokenizes. So
   verify it on the TOKENS the trainer actually reads, not on the jsonl.
2. template_unreachable - render one conversation with the grader's own
   templates/gemma3.jinja and compare byte-for-byte with the string the eval
   actually sent vLLM (recovered from the exp-01 inspect log).
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys

import numpy as np
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
NPZ = "/home/ben/task/data/sft_v1_tokens.npz"

report: dict = {}
tmpl = open(TEMPLATE).read()
report["template_sha256"] = hashlib.sha256(tmpl.encode()).hexdigest()

tok = AutoTokenizer.from_pretrained(SNAP)
tok.chat_template = tmpl
stop_id = tok.convert_tokens_to_ids("<end_of_turn>")
report["stop_token"] = "<end_of_turn>"
report["stop_token_id"] = stop_id

z = np.load(NPZ)
flat, offs, plens = z["flat"], z["offsets"], z["prompt_lens"]
n = len(plens)
last = flat[offs[1:] - 1]
report["n_rows"] = int(n)
report["rows_ending_in_stop_token"] = int((last == stop_id).sum())
report["rows_not_ending_in_stop_token"] = int((last != stop_id).sum())

lens = np.diff(offs)
report["max_row_tokens"] = int(lens.max())
report["rows_over_max_seq_len_2560"] = int((lens > 2560).sum())
report["loss_tokens_min"] = int((lens - plens).min())

# ---- 2. template reachability ------------------------------------------
log = sorted(glob.glob("/home/ben/task/logs/*gsm8k*.json"))[-1]
sample = json.load(open(log))["samples"][0]
graded_msgs = [
    {"role": m["role"], "content": m["content"]}
    for m in sample["messages"] if m["role"] in ("system", "user")
]
grader_prompt = tok.apply_chat_template(
    graded_msgs, tokenize=False, add_generation_prompt=True
)

# rebuild the same user turn from the trainer's own template constant and the
# raw question, and check it reproduces what the grader actually sent
sys.path.insert(0, "/home/ben/task")
from build_data import MATH_PROMPT_TEMPLATE  # noqa: E402

question = sample["input"]
train_user = MATH_PROMPT_TEMPLATE.format(prompt=question)
grader_user = [m for m in graded_msgs if m["role"] == "user"][-1]["content"]
report["train_user_turn_equals_grader_user_turn"] = train_user == grader_user
train_msgs = [m for m in graded_msgs if m["role"] == "system"] + [
    {"role": "user", "content": train_user}
]
train_prompt = tok.apply_chat_template(
    train_msgs, tokenize=False, add_generation_prompt=True
)
report["grader_vs_train_render_identical"] = grader_prompt == train_prompt
report["render_head"] = grader_prompt[:120]
report["render_tail"] = grader_prompt[-80:]
report["render_ends_with_model_turn"] = grader_prompt.endswith("<start_of_turn>model\n")

# a real training row, decoded, so the target's tail is visible
i = int(np.argmax(plens))          # the row with the longest prompt = a 10-shot row
row = flat[offs[i]:offs[i + 1]]
report["longest_prompt_row"] = {
    "index": i,
    "prompt_tokens": int(plens[i]),
    "total_tokens": int(len(row)),
    "prompt_head": tok.decode(row[:60]),
    "target_tail": tok.decode(row[-40:]),
}

json.dump(report, open("/home/ben/task/analysis/render_check.json", "w"), indent=2)
print(json.dumps(report, indent=2))
ok = (report["rows_not_ending_in_stop_token"] == 0
      and report["rows_over_max_seq_len_2560"] == 0
      and report["grader_vs_train_render_identical"]
      and report["render_ends_with_model_turn"])
print("VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
