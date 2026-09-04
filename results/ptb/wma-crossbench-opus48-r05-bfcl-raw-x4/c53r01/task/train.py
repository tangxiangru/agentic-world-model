#!/usr/bin/env python3
import json, os, argparse, random
import numpy as np
import torch
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

BASE = os.environ["PTB_BASE_MODEL_SNAPSHOT"]

def build_examples(tk, path, max_len):
    recs = [json.loads(l) for l in open(path).read().splitlines() if l.strip()]
    exs = []
    dropped = 0
    for r in recs:
        prompt = tk.apply_chat_template(r["messages"][:1], tools=r["tools"],
                                        tokenize=False, add_generation_prompt=True)
        full = tk.apply_chat_template(r["messages"], tools=r["tools"],
                                      tokenize=False, add_generation_prompt=False)
        comp = full[len(prompt):]
        p_ids = tk(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tk(comp, add_special_tokens=False)["input_ids"]
        ids = p_ids + c_ids
        if len(ids) > max_len:
            dropped += 1
            continue
        labels = [-100] * len(p_ids) + list(c_ids)
        exs.append({"input_ids": ids, "labels": labels, "length": len(ids)})
    print(f"built {len(exs)} examples, dropped {dropped} (> {max_len} tok)")
    return exs

class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"]); pad = maxlen - n
            input_ids.append(b["input_ids"] + [self.pad_id]*pad)
            labels.append(b["labels"] + [-100]*pad)
            attn.append([1]*n + [0]*pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lora_out")
    ap.add_argument("--data", default="train_records.jsonl")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--ga", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tk = AutoTokenizer.from_pretrained(BASE)
    tk.chat_template = open("templates/gemma3_tool_calling.jinja").read()

    exs = build_examples(tk, args.data, args.max_len)
    if args.limit:
        exs = exs[:args.limit]
    from datasets import Dataset
    ds = Dataset.from_list(exs)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        dataloader_num_workers=4,
        group_by_length=True,
        length_column_name="length",
        optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tk.pad_token_id))
    trainer.train()
    model.save_pretrained(args.out)
    tk.save_pretrained(args.out)
    print("saved LoRA adapter to", args.out)

if __name__ == "__main__":
    main()
