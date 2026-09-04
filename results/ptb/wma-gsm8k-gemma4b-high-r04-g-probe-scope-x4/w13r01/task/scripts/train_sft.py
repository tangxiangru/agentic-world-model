#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style chains.

Loss is computed on the completion only: the prompt (which for ~8% of rows
carries the harness's 2044-token 10-shot block) is masked to -100, so the model
is never trained to reproduce the few-shot examples themselves.

The vision tower and the multimodal projector are frozen and left bit-identical,
so the saved checkpoint keeps the Gemma3ForConditionalGeneration architecture
the grader's vLLM expects.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")


class CompletionOnlyDataset(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        self.rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                self.rows.append(json.loads(line))
        self.tok = tok
        self.max_len = max_len
        self.lengths = [min(r["n_tokens"], max_len) for r in self.rows]
        self.n_truncated = sum(r["n_tokens"] > max_len for r in self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = self.tok(r["prompt"], add_special_tokens=False).input_ids
        c = self.tok(r["completion"], add_special_tokens=False).input_ids
        ids = (p + c)[: self.max_len]
        labels = ([-100] * len(p) + c)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            lab.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--liger", type=int, default=1)
    ap.add_argument("--gckpt", type=int, default=1)
    args = ap.parse_args()

    if args.liger:
        # Gemma-3's 262k vocabulary makes the materialised logits the single
        # largest allocation (bf16 logits + fp32 upcast for the loss). Liger's
        # fused linear cross-entropy never materialises them.
        from liger_kernel.transformers import monkey_patch as _lk
        _lk.apply_liger_kernel_to_gemma3(
            rope=True, cross_entropy=False, fused_linear_cross_entropy=True,
            rms_norm=True, geglu=True,
        )
        print("liger: fused linear CE enabled for gemma3", flush=True)

    set_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.parent)
    ds = CompletionOnlyDataset(args.data, tok, args.max_len, args.limit)
    print(f"rows {len(ds)}  truncated {ds.n_truncated} "
          f"({ds.n_truncated / max(len(ds), 1):.4%})", flush=True)
    assert ds.n_truncated / max(len(ds), 1) < 0.02, "more than 2% of rows truncate"

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # The greedy decode file we write for vLLM (do_sample false + temperature 0.0,
    # and previously top_k -1) is rejected by GenerationConfig.save_pretrained's
    # strict validation. If the parent is one of our own checkpoints the model
    # carries that config in memory and Trainer.save_model would raise at the end
    # of training. Replace it with a valid one; the vLLM decode file is written
    # separately below, after the save.
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid",
    )

    frozen = 0
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {trainable/1e9:.3f}B  frozen {frozen/1e6:.1f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        gradient_checkpointing=bool(args.gckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        optim=args.optim,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    # group_by_length needs the lengths without touching the dataset object
    targs.length_column_name = "length"

    class LenTrainer(Trainer):
        def _get_train_sampler(self, *a, **k):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=ds.lengths,
                generator=torch.Generator().manual_seed(self.args.seed),
            )

    trainer = LenTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # decode config for the grader: evaluate.py passes no temperature, so vLLM
    # falls back to this file. Greedy, and both terminators kept.
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "do_sample": False, "temperature": 0.0, "top_p": 1.0,
                   "cache_implementation": "hybrid"}, f, indent=2)
    for extra in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.parent, extra)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, extra)):
            with open(src) as a, open(os.path.join(final, extra), "w") as b:
                b.write(a.read())
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
