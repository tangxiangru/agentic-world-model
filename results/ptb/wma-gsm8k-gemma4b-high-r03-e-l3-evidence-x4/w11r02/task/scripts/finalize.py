"""Copy a checkpoint into final_model/ and check the grader can actually load it.

pitfalls.yaml: final_model_not_loadable. The grader loads final_model/ with
vLLM from a fresh process, picks templates/gemma3.jinja by looking at
config.json's architectures, and takes its decode defaults from
generation_config.json. All three are checked here, plus a CPU load with
transformers and a tokenizer round-trip on <end_of_turn> (id 106).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REQUIRED = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    ap.add_argument("--snapshot", required=True, help="base snapshot, for aux files")
    ap.add_argument("--no-copy", action="store_true")
    args = ap.parse_args()

    if not args.no_copy:
        if os.path.exists(args.dst):
            shutil.rmtree(args.dst)
        shutil.copytree(args.src, args.dst)
        for extra in ("preprocessor_config.json", "processor_config.json",
                      "added_tokens.json", "tokenizer.model"):
            s = os.path.join(args.snapshot, extra)
            d = os.path.join(args.dst, extra)
            if os.path.exists(s) and not os.path.exists(d):
                shutil.copy2(s, d)

    problems = []
    for f in REQUIRED:
        if not os.path.exists(os.path.join(args.dst, f)):
            problems.append(f"missing {f}")

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0].lower()
    if "gemma" not in arch:
        problems.append(f"evaluate.py would not select gemma3.jinja for {arch}")

    gen = json.load(open(os.path.join(args.dst, "generation_config.json")))
    eos = gen.get("eos_token_id")
    if not (isinstance(eos, list) and 106 in eos):
        problems.append(f"generation_config eos_token_id={eos} does not contain 106")
    print("generation_config:", json.dumps(gen))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.dst)
    if tok.convert_tokens_to_ids("<end_of_turn>") != 106:
        problems.append("<end_of_turn> is not token 106 in this tokenizer")

    # render one graded prompt with the grader's own template
    import fmt
    p = fmt.render_prompt("Bob has 3 apples and buys 4 more. How many?", 10, tok)
    if not p.endswith("<start_of_turn>model\n"):
        problems.append("rendered prompt does not end with the generation prompt")
    print(f"10-shot prompt tokens: {len(tok(p, add_special_tokens=False)['input_ids'])}")

    import torch
    from transformers import Gemma3ForConditionalGeneration
    m = Gemma3ForConditionalGeneration.from_pretrained(
        args.dst, dtype=torch.bfloat16, device_map="cpu"
    )
    n = sum(x.numel() for x in m.parameters())
    print(f"loaded on CPU: {type(m).__name__}, {n/1e9:.2f}B params")

    print("PROBLEMS:", problems if problems else "none")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
