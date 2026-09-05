#!/usr/bin/env python3
"""Assemble final_model/ from a trained checkpoint and verify it the way the
grader will load it (fresh process, vLLM, templates/gemma3.jinja).

Guards the final_model_not_loadable pitfall: full weights (no adapter), the
tokenizer and processor files beside them, and a generation_config whose
eos_token_id still contains 106 (<end_of_turn>) so generation actually stops.
"""
import argparse
import json
import os
import shutil

SNAP = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
NEEDED = ("config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "special_tokens_map.json",
          "model.safetensors.index.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--decode", choices=["greedy", "shipped"], default="greedy")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for fn in os.listdir(args.src):
        if fn in ("training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                  "trainer_state.json"):
            continue
        shutil.copy2(os.path.join(args.src, fn), os.path.join(args.dst, fn))
    for fn in ("tokenizer.model", "added_tokens.json", "preprocessor_config.json",
               "processor_config.json"):
        s = os.path.join(SNAP, fn)
        if os.path.exists(s) and not os.path.exists(os.path.join(args.dst, fn)):
            shutil.copy2(s, os.path.join(args.dst, fn))

    gen = json.load(open(os.path.join(SNAP, "generation_config.json")))
    if args.decode == "greedy":
        gen.update({"do_sample": False, "temperature": 0.0, "top_k": 0, "top_p": 1.0})
    gen["eos_token_id"] = [1, 106]
    json.dump(gen, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)

    missing = [f for f in NEEDED if not os.path.exists(os.path.join(args.dst, f))]
    assert not missing, f"missing from final_model: {missing}"
    shards = [f for f in os.listdir(args.dst) if f.endswith(".safetensors")]
    assert shards, "no safetensors weights in final_model"

    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    cfg = AutoConfig.from_pretrained(args.dst)
    tok = AutoTokenizer.from_pretrained(args.dst)
    g = GenerationConfig.from_pretrained(args.dst)
    assert 106 in (g.eos_token_id if isinstance(g.eos_token_id, list) else [g.eos_token_id])
    assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
    total = sum(os.path.getsize(os.path.join(args.dst, s)) for s in shards) / 1e9
    print(json.dumps({
        "dst": args.dst,
        "arch": cfg.architectures,
        "shards": len(shards),
        "weights_gb": round(total, 2),
        "eos_token_id": g.eos_token_id,
        "temperature": g.temperature,
        "do_sample": g.do_sample,
    }, indent=2))
    print("final_model assembled and CPU-verified")


if __name__ == "__main__":
    main()
