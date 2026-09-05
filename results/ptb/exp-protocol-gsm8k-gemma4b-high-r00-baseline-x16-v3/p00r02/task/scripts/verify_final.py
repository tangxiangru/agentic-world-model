#!/usr/bin/env python3
"""Pre-deadline check on final_model/ (pitfall: final_model_not_loadable).

Loads the directory on CPU with transformers exactly as a fresh process would,
confirms the tokenizer and the extra processor files are present, prints the
decoding defaults vLLM will inherit, and renders one grader-shaped prompt so a
template regression would be visible.
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

FINAL = "/home/ben/task/final_model"
REQUIRED = ["config.json", "generation_config.json", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def main() -> None:
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(FINAL, f))]
    print("missing required files:", missing or "none")
    assert not missing

    cfg = json.load(open(os.path.join(FINAL, "config.json")))
    print("architectures:", cfg["architectures"], "| dtype:", cfg.get("dtype", cfg.get("torch_dtype")))

    gen = json.load(open(os.path.join(FINAL, "generation_config.json")))
    print("generation_config:", gen)
    assert gen.get("do_sample") is False and gen.get("temperature") == 0.0, \
        "final_model must decode greedily: vLLM inherits these as request defaults"
    assert 106 in gen["eos_token_id"], "<end_of_turn> must remain a stop token"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(FINAL)
    print("eos:", tok.eos_token, "| <end_of_turn> id:", tok.convert_tokens_to_ids("<end_of_turn>"))

    model = AutoModelForCausalLM.from_pretrained(FINAL, dtype=torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded on CPU: {n / 1e9:.2f}B params")

    prompt, target = fmt.build_example("A shop sells 47 pens at $3 each. How much?",
                                       "47 * 3 = 141.\n\nANSWER: 141")
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    print("rendered prompt tail:", repr(prompt[-90:]))
    print("prompt tokens:", len(ids), "| target tail:", repr(target[-30:]))
    print("template sha256:", fmt.template_sha256())
    print("OK")


if __name__ == "__main__":
    main()
