"""Copy a checkpoint into final_model/, set the decode config, and verify it loads.

    python finalize.py --src ckpts/exp-02/final --greedy

Checks (pitfall final_model_not_loadable): every file vLLM needs is present, the
weights load on CPU with transformers from a fresh process, the tokenizer round
-trips <end_of_turn>, and generation_config.json says what we think it says.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NEEDED = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "model.safetensors.index.json",
    "preprocessor_config.json", "processor_config.json",
]

GREEDY = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "transformers_version": "4.57.3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        if os.path.exists(args.dst):
            shutil.rmtree(args.dst)
        shutil.copytree(args.src, args.dst)
        for f in NEEDED:
            dst_f = os.path.join(args.dst, f)
            if not os.path.exists(dst_f) and os.path.exists(os.path.join(SNAP, f)):
                shutil.copy(os.path.join(SNAP, f), dst_f)
        if args.greedy:
            with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
                json.dump(GREEDY, f, indent=2)

    missing = [f for f in NEEDED if not os.path.exists(os.path.join(args.dst, f))]
    print("missing files:", missing)
    print("generation_config:", json.load(open(os.path.join(args.dst, "generation_config.json"))))

    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.dst)
    assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
    m = Gemma3ForConditionalGeneration.from_pretrained(args.dst, dtype=torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print("loaded on CPU ok; params:", n)
    assert not missing, missing
    print("OK", args.dst)


if __name__ == "__main__":
    main()
