#!/usr/bin/env python3
"""CPU dry run: does what we train on render exactly like what the grader sends?

1. Render one training row with the grader's own templates/gemma3.jinja and
   compare byte-for-byte with scripts/common.render_prompt.
2. Rebuild the grader's 10-shot system message (inspect_evals gsm8k,
   fewshot_seed=42) and report its token length, then report the token-length
   distribution of the training rows against --max-seq-len.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TEMPLATE_PATH, render_prompt, render_target  # noqa: E402

from jinja2 import Environment  # noqa: E402
from jinja2.exceptions import TemplateError  # noqa: E402


def jinja_render(messages, add_generation_prompt=True):
    with open(TEMPLATE_PATH) as f:
        src = f.read()
    env = Environment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg):
        raise TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    t = env.from_string(src)
    return t.render(
        messages=messages, bos_token="<bos>", add_generation_prompt=add_generation_prompt
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-seq-len", type=int, required=True)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    with open(TEMPLATE_PATH, "rb") as f:
        print("template sha256:", hashlib.sha256(f.read()).hexdigest())

    rows = []
    with open(args.data) as f:
        for line in f:
            rows.append(json.loads(line))

    # --- 1. rendering equality -------------------------------------------
    bad = 0
    for r in rows[:200]:
        ours = render_prompt(r["prompt"])
        theirs = jinja_render([{"role": "user", "content": r["prompt"]}])
        if ours != theirs:
            bad += 1
            if bad == 1:
                print("MISMATCH\n--ours--\n%r\n--jinja--\n%r" % (ours[-300:], theirs[-300:]))
    print(f"render check: {200 - bad}/200 identical to templates/gemma3.jinja")

    # also check the system-message folding path the grader actually uses
    sysmsg = "EX1\n\nEX2"
    a = jinja_render(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": "Q"}]
    )
    b = render_prompt(sysmsg + "\n\n" + "Q")
    print(f"system-folding check: {'identical' if a == b else 'MISMATCH'}")
    if a != b:
        print(repr(a))
        print(repr(b))

    # --- 2. the grader's real 10-shot prefix ------------------------------
    from inspect_evals.gsm8k.gsm8k import sample_to_fewshot  # noqa: E402
    from inspect_ai.dataset import hf_dataset  # noqa: E402
    from inspect_evals.gsm8k.gsm8k import record_to_sample  # noqa: E402

    fewshots = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=42,
        limit=10,
    )
    prefix = "\n\n".join(sample_to_fewshot(s) for s in fewshots)

    from transformers import AutoTokenizer  # noqa: E402

    tok = AutoTokenizer.from_pretrained(args.model)
    n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
    print(f"grader 10-shot prefix: {n_prefix} tokens")
    with open(os.path.join(os.path.dirname(args.data), "_eval_fewshot_prefix.txt"), "w") as f:
        f.write(prefix)

    # --- 3. training-row token lengths ------------------------------------
    import numpy as np  # noqa: E402

    sub = rows[: args.n]
    plens, clens = [], []
    for r in sub:
        p = tok(render_prompt(r["prompt"]), add_special_tokens=False)["input_ids"]
        c = tok(render_target(r["completion"]), add_special_tokens=False)["input_ids"]
        plens.append(len(p))
        clens.append(len(c))
    tot = np.array(plens) + np.array(clens)
    print(
        "prompt tokens  p50=%d p95=%d max=%d" % tuple(np.percentile(plens, [50, 95]).tolist() + [max(plens)])
    )
    print(
        "target tokens  p50=%d p95=%d max=%d" % tuple(np.percentile(clens, [50, 95]).tolist() + [max(clens)])
    )
    print(
        "total  tokens  p50=%d p95=%d p99=%d max=%d"
        % tuple(np.percentile(tot, [50, 95, 99]).tolist() + [tot.max()])
    )
    trunc = float((tot > args.max_seq_len).mean())
    print(f"rows longer than max_seq_len={args.max_seq_len}: {trunc:.4%}")
    print(f"mean tokens/row: {tot.mean():.1f}  -> {tot.mean() * len(rows) / 1e6:.1f}M tokens/epoch")

    # --- 4. stop token / answer marker in the data -------------------------
    from common import ANSWER_MARKER, STOP_TOKEN  # noqa: E402

    bad_stop = sum(1 for r in rows if not render_target(r["completion"]).endswith(STOP_TOKEN))
    bad_marker = sum(1 for r in rows if r["completion"].count(ANSWER_MARKER) != 1)
    hashes = sum(1 for r in rows if "####" in r["completion"])
    print(f"targets not ending in {STOP_TOKEN}: {bad_stop}")
    print(f"targets without exactly one '{ANSWER_MARKER}': {bad_marker}")
    print(f"targets still containing '####': {hashes}")

    # last numeric token of the completion must be the gold answer
    import re  # noqa: E402

    def last_num(s):
        words = s.strip().replace(STOP_TOKEN, "").strip().split()
        for w in reversed(words):
            if w.replace(".", "").replace(",", "").replace("$", "").isnumeric():
                return w.replace(",", "").replace("$", "")
        return None

    mism = 0
    for r in rows[:5000]:
        ln = last_num(r["completion"])
        if ln is None or ln.rstrip(".").lstrip("0") not in (
            r["answer"].lstrip("0"),
            r["answer"],
        ):
            mism += 1
    print(f"rows (of 5000) whose last numeric token != gold answer: {mism}")


if __name__ == "__main__":
    main()
