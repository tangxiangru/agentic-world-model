#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save a full model dir for vllm."""
import os, argparse, shutil, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from peft import PeftModel

BASE = os.environ["PTB_BASE_MODEL_SNAPSHOT"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = Gemma3ForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    model.save_pretrained(args.out)
    tk = AutoTokenizer.from_pretrained(BASE)
    tk.save_pretrained(args.out)
    print("merged model saved to", args.out)

if __name__ == "__main__":
    main()
