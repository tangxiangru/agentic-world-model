#!/usr/bin/env python3
"""Make a Trainer checkpoint loadable by the grader's vLLM.

Trainer's intermediate checkpoint-N/ directories hold fp32 weights and no
tokenizer, no preprocessor/processor config and a generation_config that
Trainer rewrote.  evaluate.py hands the bare directory to vLLM, so all of that
has to be put back (pitfall: final_model_not_loadable) and the weights cast to
bf16 (fp32 weights would be ~17 GB against a 24 GB budget at
--gpu-memory-utilization 0.3, leaving no KV cache).

Optionally rewrites generation_config.json to decode greedily.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--base", default=os.environ.get("PTB_BASE_MODEL_SNAPSHOT"))
    ap.add_argument("--greedy", action="store_true", help="temperature 0, no top_k/top_p")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    from transformers import AutoConfig, Gemma3ForConditionalGeneration

    cfg = AutoConfig.from_pretrained(args.src)
    cls = Gemma3ForConditionalGeneration
    if not (cfg.architectures and "ConditionalGeneration" in cfg.architectures[0]):
        from transformers import AutoModelForCausalLM

        cls = AutoModelForCausalLM
    model = cls.from_pretrained(args.src, dtype=torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    model.config.use_cache = True
    model.save_pretrained(args.dst, safe_serialization=True)

    # copy the tokenizer files verbatim from the immutable snapshot rather than
    # re-serialising them: transformers 4.57 rewrites tokenizer.json into a form
    # that warns about the regex pattern on reload.  The vocabulary is unchanged
    # by fine-tuning, so the original bytes are the correct ones to ship.
    for fname in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                  "special_tokens_map.json", "added_tokens.json",
                  "preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.base, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.dst, fname))

    gen = json.load(open(os.path.join(args.base, "generation_config.json")))
    if args.greedy:
        gen.pop("top_k", None)
        gen.pop("top_p", None)
        gen["do_sample"] = False
        gen["temperature"] = 0.0
    json.dump(gen, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)

    print(json.dumps({"dst": args.dst, "generation_config": gen}, indent=2))
    print("files:", sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
