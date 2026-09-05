"""Package a trained checkpoint as final_model/ and verify it loads.

Guards the two pitfalls that turn a good run into a zero:
  * eos_mismatch      - generation_config.eos_token_id must still contain 106
                        (<end_of_turn>), the token every training target ends on.
  * final_model_not_loadable - the directory must carry weights + tokenizer +
                        processor and must load from a fresh process.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NEEDED = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "preprocessor_config.json", "processor_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    ap.add_argument("--greedy", action="store_true",
                    help="pin temperature 0 in generation_config (vLLM reads it as the server default)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst,
                    ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt", "rng_state*",
                                                  "trainer_state.json", "training_args.bin", "*.tmp"))
    for f in NEEDED:
        d = os.path.join(args.dst, f)
        if not os.path.exists(d) and os.path.exists(os.path.join(SNAP, f)):
            shutil.copy(os.path.join(SNAP, f), d)
            print("copied missing", f, "from the base snapshot")

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path))
    eos = gc.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    if 106 not in eos:
        eos = sorted(set(eos + [1, 106]))
        print("FIXED eos_token_id ->", eos)
    gc["eos_token_id"] = eos
    if args.greedy:
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc.pop("top_k", None)
        gc.pop("top_p", None)
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("generation_config:", json.dumps(gc))

    print(sorted(os.listdir(args.dst)))

    if args.verify:
        import torch
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
        tok = AutoTokenizer.from_pretrained(args.dst)
        m = Gemma3ForConditionalGeneration.from_pretrained(args.dst, dtype=torch.bfloat16,
                                                           device_map="cpu")
        print("loaded ok:", type(m).__name__, sum(p.numel() for p in m.parameters()) / 1e9, "B params")
        print("eos in model gen config:", m.generation_config.eos_token_id)
        print("tokenizer <end_of_turn> ->", tok.convert_tokens_to_ids("<end_of_turn>"))


if __name__ == "__main__":
    main()
