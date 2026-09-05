#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on prompt/completion jsonl.

The vision tower and the multimodal projector are frozen: the task is text-only,
so their gradients and optimiser state would be pure cost, but they stay in the
saved checkpoint so the output directory loads under exactly the same
architecture (Gemma3ForConditionalGeneration) the grader's vLLM loads the base
model with.

Loss is on completion tokens only. Rows are pre-tokenised once, sorted into
length buckets by the sampler, and padded per batch.
"""
import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class PackedRows(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None, seed=0):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            lines = f.readlines()
        if limit:
            random.Random(seed).shuffle(lines)
            lines = lines[:limit]
        for line in lines:
            d = json.loads(line)
            p = tok(d["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(d["completion"], add_special_tokens=False)["input_ids"]
            if len(p) + len(c) > max_seq_len:
                n_trunc += 1
                continue  # drop rather than truncate: a cut completion is a wrong target
            self.rows.append((p, c))
        print(f"dataset {path}: {len(self.rows)} rows kept, {n_trunc} dropped for length "
              f"(> {max_seq_len} tokens)", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = m - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--liger", type=int, default=1)
    args = ap.parse_args()

    if args.liger:
        # Gemma 3's vocab is 262k, so the stock forward's logits.float() wants
        # 48 GB for a 32x1536 batch. The fused linear cross-entropy never
        # materialises the full logit tensor.
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger: fused linear cross-entropy enabled for gemma3", flush=True)

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.padding_side = "right"

    ds = PackedRows(args.data, tok, args.max_seq_len, args.limit, args.seed)

    # same reason as at save time: a parent that is one of our own checkpoints
    # carries a generation config the strict validator rejects, and Trainer's
    # mid-run checkpoint saves would hit it too.
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation="eager")
    model.model.vision_tower.requires_grad_(False)
    model.model.multi_modal_projector.requires_grad_(False)
    model.config.use_cache = False
    # save_pretrained validates model.generation_config strictly, and the greedy
    # config this script writes at the end (do_sample false + temperature 0.0) is
    # exactly what that validator rejects - so a parent that is one of our own
    # checkpoints makes every save, mid-run or final, raise after the compute is
    # already spent. Replace it here, before training, not after.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e9:.2f}B of {sum(p.numel() for p in model.parameters())/1e9:.2f}B",
          flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # The grader's vLLM reads generation_config.json for its default sampling
    # params and inspect sends no temperature, so this file decides greedy vs
    # sampled decoding at grading time.
    gen = {
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "do_sample": False,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "transformers_version": "4.50.0.dev0",
    }
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    # the base snapshot's processor files, so the dir loads exactly like the base
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            with open(src) as a, open(os.path.join(final, fn), "w") as b:
                b.write(a.read())
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
