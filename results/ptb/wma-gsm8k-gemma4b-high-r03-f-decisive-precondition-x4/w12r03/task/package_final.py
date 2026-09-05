"""Package a checkpoint as final_model/: bf16 weights + tokenizer + greedy config.

The grader loads final_model/ with vLLM from a fresh process, so everything it
needs must be in the directory: config.json, weights, tokenizer files, the
processor config gemma-3 ships, and a generation_config.json that decodes
greedily (vLLM reads its default sampling params from there).
"""
import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from train_sft import GEN_CONFIG

TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "tokenizer.model", "preprocessor_config.json",
             "processor_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkpoint dir with the weights")
    ap.add_argument("--tok", required=True, help="dir holding the tokenizer/processor files")
    ap.add_argument("--out", default="/home/ben/task/final_model")
    args = ap.parse_args()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    model = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype=torch.bfloat16)
    model.config.use_cache = True
    # the source checkpoint's greedy generation_config fails strict validation on
    # save (do_sample False with temperature 0.0); the greedy JSON is written
    # explicitly below, so neutralise the in-memory copy first
    for f in ("temperature", "top_k", "top_p"):
        setattr(model.generation_config, f, None)
    model.save_pretrained(args.out, safe_serialization=True)
    del model

    AutoTokenizer.from_pretrained(args.tok).save_pretrained(args.out)
    try:
        AutoProcessor.from_pretrained(args.tok).save_pretrained(args.out)
    except Exception as e:
        print("processor save failed:", e)
    for f in TOK_FILES:
        s = os.path.join(args.tok, f)
        if os.path.exists(s) and not os.path.exists(os.path.join(args.out, f)):
            shutil.copy(s, args.out)

    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(GEN_CONFIG, f, indent=2)

    print("wrote", args.out)
    print(sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
