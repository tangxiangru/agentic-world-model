#!/usr/bin/env python3
"""Full fine-tune the language model of gemma-3-4b-pt on GSM8K-style SFT data."""
import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="train_gsm8k.jsonl")
    p.add_argument("--out", type=str, default="sft_out")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--maxlen", type=int, default=768)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--init_from", type=str, default=SNAP,
                   help="model to initialize from (SNAP for base, or a checkpoint dir)")
    return p.parse_args()


def main():
    args = parse_args()
    with open("templates/gemma3.jinja") as f:
        chat_template = f.read()

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template = chat_template
    if tok.pad_token is None:
        tok.pad_token = tok.convert_ids_to_tokens(0)

    EOT = tok.convert_tokens_to_ids("<end_of_turn>")

    def encode(example):
        msgs = example["messages"]
        # prompt = everything up to and including "<start_of_turn>model\n"
        prompt_text = tok.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        # truncate
        full_ids = full_ids[: args.maxlen]
        labels = list(full_ids)
        n_prompt = min(len(prompt_ids), len(full_ids))
        for i in range(n_prompt):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(encode, remove_columns=ds.column_names, num_proc=8)
    # filter out any example where all labels are masked (too long)
    ds = ds.filter(lambda ex: any(l != -100 for l in ex["labels"]), num_proc=8)
    print(f"training examples: {len(ds)}")
    lens = [len(x) for x in ds["input_ids"]]
    print(f"len stats: max={max(lens)} mean={sum(lens)/len(lens):.0f}")

    print(f"loading model from {args.init_from} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.init_from,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    # freeze vision tower + multimodal projector (unused for text GSM8K)
    frozen, trainable = 0, 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
        else:
            trainable += p.numel()
    print(f"trainable params: {trainable/1e9:.2f}B  frozen: {frozen/1e9:.2f}B")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    collator = DataCollatorForSeq2Seq(
        tok, model=model, label_pad_token_id=-100, padding=True
    )

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        optim="adamw_torch",
        report_to=[],
        gradient_checkpointing=False,  # enabled manually above
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()

    print(f"saving to {args.out} ...")
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("done.")


if __name__ == "__main__":
    main()
