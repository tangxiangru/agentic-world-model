#!/usr/bin/env python3
"""Completion-only SFT with LEFT-truncation of the prompt.

Same as train_sft.py but when prompt+completion exceeds max_len, tokens are
dropped from the LEFT of the PROMPT (earliest few-shot examples), never from the
completion -- so every row keeps full loss signal on its answer (avoids the
seq_len truncation pitfall for long few-shot rows).
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
            if len(p) + len(c) > max_len:
                n_trunc += 1
                keep_p = max(0, max_len - len(c))
                p = p[-keep_p:] if keep_p > 0 else []
                c = c[:max_len]            # only if completion alone > max_len
            ids = p + c
            labels = [-100] * len(p) + list(c)
            self.rows.append((ids, labels))
        print(f"loaded {len(self.rows)} rows, {n_trunc} left-truncated "
              f"({100*n_trunc/max(1,len(self.rows)):.2f}%) [completion preserved]")

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
            n = len(b["input_ids"]); pad = m - n
            ii.append(b["input_ids"] + [self.pad_id] * pad)
            ll.append(b["labels"] + [-100] * pad)
            am.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(args.template).read()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    ds = JsonlSFT(args.data, tok, args.max_len)
    collator = Collator(tok.pad_token_id)
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=args.warmup,
        weight_decay=args.wd, bf16=True, logging_steps=10, save_strategy="no",
        report_to=[], seed=args.seed, dataloader_num_workers=2, optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
