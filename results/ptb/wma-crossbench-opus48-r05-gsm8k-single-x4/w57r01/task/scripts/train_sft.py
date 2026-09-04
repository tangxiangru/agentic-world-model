#!/usr/bin/env python3
"""Completion-only SFT of gemma-3-4b-pt on GSM8K-train-derived CoT data.

Loads the immutable snapshot, freezes the vision tower, trains the language
model on (prompt, completion) pairs with the prompt masked out of the loss.
Targets end in <end_of_turn> (id 106), which is in the model's eos list and is
the terminator the gemma3.jinja grader template stops on.
"""
import argparse, json, os
import torch
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)
from torch.utils.data import Dataset


class JsonlSFT(Dataset):
    def __init__(self, path, tok, max_len):
        self.rows = []
        n_trunc = 0
        for line in open(path):
            r = json.loads(line)
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"]
            ids = p + c
            labels = [-100] * len(p) + list(c)
            if len(ids) > max_len:
                n_trunc += 1
                ids = ids[:max_len]
                labels = labels[:max_len]
            self.rows.append((ids, labels))
        print(f"loaded {len(self.rows)} rows, {n_trunc} truncated "
              f"({100*n_trunc/max(1,len(self.rows)):.2f}%)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ii, ll, am = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = m - n
            ii.append(b["input_ids"] + [self.pad_id] * pad)
            ll.append(b["labels"] + [-100] * pad)
            am.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(ii),
                "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(args.template).read()

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    # freeze vision tower + projector (text-only training)
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    ds = JsonlSFT(args.data, tok, args.max_len)
    collator = Collator(tok.pad_token_id)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator)
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
