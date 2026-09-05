#!/usr/bin/env python3
import os, json, argparse
import torch
from transformers import (AutoTokenizer, AutoModelForImageTextToText,
                          Trainer, TrainingArguments)
from datasets import load_dataset

BASE = os.environ["PTB_BASE_MODEL_SNAPSHOT"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train_sft.jsonl")
    p.add_argument("--out", default="sft_out")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--maxlen", type=int, default=1024)
    p.add_argument("--warmup", type=float, default=0.03)
    return p.parse_args()


def main():
    args = parse()
    tok = AutoTokenizer.from_pretrained(BASE)

    def build(ex):
        # match eval rendering exactly: <bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n{c}<end_of_turn>\n
        prompt = f"<start_of_turn>user\n{ex['prompt']}<end_of_turn>\n<start_of_turn>model\n"
        completion = f"{ex['completion']}<end_of_turn>\n"
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        p_ids = [tok.bos_token_id] + p_ids
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        input_ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + c_ids[:]
        input_ids = input_ids[:args.maxlen]
        labels = labels[:args.maxlen]
        return {"input_ids": input_ids, "labels": labels}

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(build, remove_columns=ds.column_names, num_proc=8)
    lens = [len(x) for x in ds["input_ids"]]
    print(f"n={len(ds)} maxlen_seen={max(lens)} mean={sum(lens)/len(lens):.0f} "
          f">maxlen={sum(l>=args.maxlen for l in lens)}")

    class Collator:
        def __call__(self, feats):
            maxl = max(len(f["input_ids"]) for f in feats)
            ids, lbl, att = [], [], []
            for f in feats:
                pad = maxl - len(f["input_ids"])
                ids.append(f["input_ids"] + [tok.pad_token_id] * pad)
                lbl.append(f["labels"] + [-100] * pad)
                att.append([1] * len(f["input_ids"]) + [0] * pad)
            return {"input_ids": torch.tensor(ids),
                    "labels": torch.tensor(lbl),
                    "attention_mask": torch.tensor(att)}

    model = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    # freeze vision tower + multimodal projector; train language model only
    ntrain, nfreeze = 0, 0
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)
            nfreeze += p.numel()
        else:
            ntrain += p.numel()
    print(f"trainable={ntrain/1e9:.2f}B frozen={nfreeze/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=4,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator())
    trainer.train()
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
