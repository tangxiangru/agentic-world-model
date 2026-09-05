#!/usr/bin/env python3
"""Last-line check on final_model/ before the deadline (pitfall final_model_not_loadable).

Loads the directory from a fresh process with transformers, renders one dev-style
prompt through the grader's own template, generates, and asserts the completion
terminates on <end_of_turn> with exactly one 'ANSWER: ' line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

Q = ("A bakery sells cupcakes in boxes of 8. On Monday it baked 96 cupcakes and sold "
     "7 full boxes. On Tuesday it baked 40 more and sold 5 full boxes. How many "
     "cupcakes does the bakery have left?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="final_model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("== files:", sorted(os.listdir(args.path)))
    cfg = json.load(open(os.path.join(args.path, "config.json")))
    print("== architectures:", cfg["architectures"], "| root dtype:", cfg.get("dtype"))
    gc = json.load(open(os.path.join(args.path, "generation_config.json")))
    print("== generation_config:", gc)
    assert 106 in gc["eos_token_id"], "eos_token_id lost <end_of_turn> (106)"

    tok = AutoTokenizer.from_pretrained(args.path)
    m = Gemma3ForConditionalGeneration.from_pretrained(
        args.path, dtype=torch.bfloat16).to(args.device).eval()

    prompt = render.render_prompt(tok, Q)
    assert prompt.startswith("<bos><start_of_turn>user"), prompt[:60]
    assert prompt.endswith("<start_of_turn>model\n"), repr(prompt[-40:])
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=512, do_sample=False)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    print("== completion:\n" + text)

    assert text.rstrip().endswith("<end_of_turn>"), "did not terminate on <end_of_turn>"
    assert text.count("ANSWER: ") == 1, f"ANSWER marker count = {text.count('ANSWER: ')}"

    from inspect_ai.scorer._common import match_str
    body = text.replace("<end_of_turn>", "").replace("<eos>", "")
    _, ok = match_str(body, "40", location="end", numeric=True)
    print("== grader would read:", match_str(body, "40", location="end", numeric=True)[0])
    print("== correct on this probe:", ok, "(gold 40)")
    print("\nfinal_model loads, renders, terminates and parses. OK.")


if __name__ == "__main__":
    main()
