#!/usr/bin/env python3
"""SFT fine-tuning of gemma-3-4b-pt on math CoT data (completion-only loss)."""
import os, argparse
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse()
    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.shuffle(seed=42)
    print("Dataset size:", len(ds))

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if ("vision_tower" in name) or ("multi_modal_projector" in name):
            p.requires_grad = False
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen: {n_frozen/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")

    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs={"num_cycles": 0.5},
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        packing=False,
        completion_only_loss=True,
        chat_template_path="templates/gemma3.jinja",
        report_to=[],
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        dataset_num_proc=8,
        seed=42,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.output)
    # ensure tokenizer + chat template saved
    tok = AutoTokenizer.from_pretrained(BASE)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()
    tok.save_pretrained(args.output)
    print("Saved to", args.output)

if __name__ == "__main__":
    main()
