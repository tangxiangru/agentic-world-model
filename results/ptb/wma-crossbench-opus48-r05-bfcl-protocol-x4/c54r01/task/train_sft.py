#!/usr/bin/env python3
"""Full SFT of gemma-3-4b-pt for BFCL-style single tool calling.
Completion-only loss. Loads the multimodal Gemma3ForConditionalGeneration so the
saved model is architecturally identical to the base snapshot (vision tower frozen
and untouched)."""
import json, argparse, os, math
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

class SFTDS(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        self.ex = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False).input_ids
                c = tok(r["completion"], add_special_tokens=False).input_ids
                ids = p + c
                if len(ids) > max_len:
                    continue
                labels = [-100]*len(p) + list(c)
                self.ex.append((ids, labels))
                if limit and len(self.ex) >= limit:
                    break
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        ids, labels = self.ex[i]
        return {"input_ids": ids, "labels": labels}

class Collator:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]; lab = b["labels"]
            pad = m - len(ids)
            input_ids.append(ids + [self.pad_id]*pad)
            labels.append(lab + [-100]*pad)
            attn.append([1]*len(ids) + [0]*pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--optim", default="paged_adamw_8bit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template = open("templates/gemma3_tool_calling.jinja").read()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    ds = SFTDS(args.data, tok, args.max_seq_len, args.limit)
    print(f"training examples: {len(ds)}")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        SNAP, dtype=torch.bfloat16, attn_implementation="sdpa")
    # freeze vision tower + projector; train language model only
    nfrozen = 0
    for n, p in model.named_parameters():
        if ("vision_tower" in n) or ("multi_modal_projector" in n):
            p.requires_grad_(False); nfrozen += 1
    print(f"frozen vision/projector params: {nfrozen}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
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
        optim=args.optim,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(pad_id))
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved model to {args.out}")

if __name__ == "__main__":
    main()
