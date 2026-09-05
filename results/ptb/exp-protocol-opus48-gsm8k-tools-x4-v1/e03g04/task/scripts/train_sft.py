#!/usr/bin/env python3
"""Full SFT of google/gemma-3-4b-pt (text LM only; vision tower frozen) on
GSM8K-train-derived data. Completion-only loss. Uses the grader's exact
gemma3.jinja template so training and evaluation render identically.
Final save wrapped in GenerationSaveContract (protocol save-safety)."""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
)

from awm.exp_protocol.save_contract import GenerationSaveContract


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--template", default="templates/gemma3.jinja")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, template, max_len):
        self.tok = tokenizer
        self.template = template
        self.max_len = max_len
        self.rows = [json.loads(l) for l in open(path)]
        self.n_trunc = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        pmsgs = []
        if r.get("system"):
            pmsgs.append({"role": "system", "content": r["system"]})
        pmsgs.append({"role": "user", "content": r["prompt"]})
        # Prompt rendered via the grader's exact template (ends with
        # "<start_of_turn>model\n"); completion already ends with <end_of_turn>.
        pstr = self.tok.apply_chat_template(
            pmsgs, tokenize=False, add_generation_prompt=True, chat_template=self.template
        )
        fstr = pstr + r["completion"]
        pids = self.tok(pstr, add_special_tokens=False).input_ids
        fids = self.tok(fstr, add_special_tokens=False).input_ids
        if len(fids) > self.max_len:
            self.n_trunc += 1
            fids = fids[: self.max_len]
        labels = [-100] * len(pids) + fids[len(pids):]
        labels = labels[: len(fids)]
        return {"input_ids": fids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]
            lab = b["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    template = open(args.template).read()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    # Freeze vision tower + projector; train the text language model only.
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad = False
            n_freeze += p.numel()
        else:
            p.requires_grad = True
            n_train += p.numel()
    print(f"trainable params: {n_train/1e9:.3f}B | frozen: {n_freeze/1e9:.3f}B", flush=True)

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    ds = SFTDataset(args.data, tokenizer, template, args.max_seq_len)
    collator = Collator(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        gradient_checkpointing=False,  # enabled manually above
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )

    # Protocol save-safety: validate generation-config serialization before compute.
    saves = GenerationSaveContract(policy="inactive_sampling_v1")
    saves.check_before_compute(model)

    trainer.train()
    print(f"rows truncated at max_seq_len={args.max_seq_len}: {ds.n_trunc}", flush=True)

    # Final save under the save contract.
    model.config.use_cache = True
    with saves.saving(model, args.out):
        model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    try:
        proc = AutoProcessor.from_pretrained(args.model)
        proc.save_pretrained(args.out)
    except Exception as e:
        print("processor save skipped:", str(e)[:120], flush=True)
    print("SAVED_OK", args.out, flush=True)


if __name__ == "__main__":
    main()
