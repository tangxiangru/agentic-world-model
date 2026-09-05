#!/usr/bin/env python3
"""Pre-deadline check on final_model/ (pitfall: final_model_not_loadable).

Loads the directory on CPU with transformers exactly as a fresh process would,
checks the tokenizer and generation_config are present and sane, renders one
grader-format prompt, and prints what evaluate.py's model_type() will resolve to.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/ben/task/scripts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/home/ben/task/final_model")
    args = ap.parse_args()
    p = args.path
    ok = True

    print("== files")
    for fn in sorted(os.listdir(p)):
        print("  ", fn)

    cfg = json.load(open(os.path.join(p, "config.json")))
    arch = cfg["architectures"][0]
    print("== architectures:", arch)
    assert "gemma" in arch.lower(), "evaluate.py's model_type() needs a gemma architecture"

    gc = json.load(open(os.path.join(p, "generation_config.json")))
    print("== generation_config:", json.dumps(gc))
    if 106 not in (gc.get("eos_token_id") or []):
        print("!! eos_token_id does not contain 106 (<end_of_turn>) -- vLLM will not stop")
        ok = False
    if gc.get("do_sample", True) is not False or gc.get("temperature", 1.0) != 0.0:
        print("!! not greedy; exp-03 measured greedy at +6.0 points")
        ok = False

    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(p)
    assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
    print("== tokenizer ok, <end_of_turn> = 106")

    from format_utils import eval_fewshot_system, render_prompt
    txt = render_prompt(tok, "Solve this.\n\nWhat is 2+2?\n\nReasoning:", eval_fewshot_system())
    print("== rendered 10-shot prompt tokens:", len(tok(txt, add_special_tokens=False)["input_ids"]))
    print("== prompt tail:", repr(txt[-80:]))

    print("== loading on CPU (this takes a minute)")
    import torch
    m = Gemma3ForConditionalGeneration.from_pretrained(p, dtype=torch.bfloat16, device_map="cpu")
    n = sum(x.numel() for x in m.parameters())
    print(f"== loaded ok, {n/1e9:.2f}B params")
    print("RESULT:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
