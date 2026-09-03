#!/usr/bin/env python3
"""CPU dry run: prove the training rows render byte-for-byte like the grader's
prompt, end in the token vLLM stops on, and fit inside max_seq_len."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    tmpl = open(TEMPLATE).read()
    print("template sha256", hashlib.sha256(tmpl.encode()).hexdigest()[:16])
    tok = AutoTokenizer.from_pretrained(SNAP)

    rows = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            rows.append(json.loads(line))

    # ---- 1. rendering: my prompt string == what the grading template produces
    q = rows[0]["question"]
    user = MATH_PROMPT_TEMPLATE.format(prompt=q)
    grader = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        chat_template=tmpl,
        tokenize=False,
        add_generation_prompt=True,
    )
    mine = tok.bos_token + rows[0]["prompt"]
    print("\n--- grader render (zero-shot) ---")
    print(repr(grader[:120]), "...", repr(grader[-80:]))
    print("--- mine ---")
    print(repr(mine[:120]), "...", repr(mine[-80:]))
    assert grader == mine, "RENDER MISMATCH"
    print("RENDER MATCH: ok")

    # with a system message (the grader's 10-shot case) the only difference is
    # the shots pasted in front of the user content
    fs = "SHOT1\n\nSHOT2"
    grader_fs = tok.apply_chat_template(
        [{"role": "system", "content": fs}, {"role": "user", "content": user}],
        chat_template=tmpl,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert grader_fs == tok.bos_token + "<start_of_turn>user\n" + fs + "\n\n" + user + "<end_of_turn>\n<start_of_turn>model\n"
    print("FEWSHOT RENDER: system content is pasted before the user content, ok")

    # ---- 2. stop token
    eot = tok("<end_of_turn>", add_special_tokens=False)["input_ids"]
    print("<end_of_turn> ->", eot, tok.convert_ids_to_tokens(eot))
    assert eot == [106], eot
    gen = json.load(open(SNAP + "/generation_config.json"))
    print("generation_config eos_token_id", gen["eos_token_id"])
    assert 106 in gen["eos_token_id"], "vLLM will not stop on <end_of_turn>"

    # ---- 3. every target ends with the stop token, marker appears once
    bad_end = bad_marker = 0
    lens_p, lens_c = [], []
    for r in rows:
        c = tok(r["completion"], add_special_tokens=False)["input_ids"]
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        if c[-1] != 106:
            bad_end += 1
        if r["completion"].count("ANSWER:") != 1:
            bad_marker += 1
        lens_p.append(len(p))
        lens_c.append(len(c))
    print(f"targets not ending in <end_of_turn>: {bad_end}")
    print(f"targets whose 'ANSWER:' count != 1: {bad_marker}")
    assert bad_end == 0 and bad_marker == 0

    tot = np.array(lens_p) + np.array(lens_c) + 1
    print(
        f"tokens/row  p50={np.percentile(tot,50):.0f} p90={np.percentile(tot,90):.0f} "
        f"p99={np.percentile(tot,99):.0f} max={tot.max()} mean={tot.mean():.0f}"
    )
    over = (tot > args.max_len).mean()
    print(f"share over max_len={args.max_len}: {over:.4f}")
    assert over <= 0.02, "more than 2% of rows would truncate"

    # ---- 4. the answer the grader would read from a target
    import re

    def graded(text: str) -> str | None:
        words = re.split(r"\s+", text.replace("<end_of_turn>", "").strip())
        for w in reversed(words):
            w2 = re.sub(r"[^0-9.\-]", "", w)
            if re.match(r"^-?\d", w2 or ""):
                return w2.rstrip(".")
        return None

    bad = sum(1 for r in rows if graded(r["completion"]) != r["answer"])
    print(f"targets where the last number != the gold answer: {bad}/{len(rows)}")
    assert bad == 0
    print("\nALL DRY-RUN CHECKS PASSED")
    print("\n--- one full training row ---")
    print(tok.bos_token + rows[0]["prompt"] + rows[0]["completion"])


if __name__ == "__main__":
    main()
