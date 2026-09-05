#!/usr/bin/env python3
"""Completion-only SFT for google/gemma-3-4b-pt on GSM8K-style data.

Rows are {"prompt", "completion"} already rendered with the grader's
chat template (templates/gemma3.jinja); this script only tokenizes,
masks the prompt out of the loss, and trains.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class SFTData(Dataset):
    def __init__(self, path: str, tok, max_len: int, limit: int | None = None):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                ids = p + c
                if len(ids) > max_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p) + c[:]
                self.rows.append((ids, labels))
        print(f"[data] {len(self.rows)} rows from {path}, dropped {n_trunc} over {max_len} tokens")
        self.lengths = [len(a) for a, _ in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(features, pad_id: int):
    n = max(len(f["input_ids"]) for f in features)
    n = ((n + 7) // 8) * 8
    input_ids, labels, mask = [], [], []
    for f in features:
        k = n - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * k)
        labels.append(f["labels"] + [-100] * k)
        mask.append([1] * len(f["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-epochs", type=int, default=1)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--liger", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTData(args.data, tok, args.max_seq_len, args.limit)

    model = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=torch.float32, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # A parent checkpoint saved by this script carries the session's greedy
    # decode config (do_sample False + temperature 0.0). transformers 4.57
    # refuses to SAVE that combination, so every checkpoint write would die
    # after the training is already done. Neutralise it on the in-memory
    # config; the greedy json is written back verbatim at the end.
    gc = model.generation_config
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    # the vision tower is never exercised by text-only GSM8K rows; freezing it
    # keeps it byte-identical to the base snapshot and saves optimizer state
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_train/1e9:.2f}B, frozen {n_frozen/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy=("steps" if args.save_steps else ("epoch" if args.save_epochs else "no")),
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        report_to=[],
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        optim=args.optim,
        seed=args.seed,
        max_grad_norm=1.0,
        # gemma-3's 262k vocab makes the fp32 logits tensor the memory wall
        # (bs16 x 2048 x 262144 x 4B = 31 GB); liger's fused linear
        # cross-entropy never materialises it.
        use_liger_kernel=args.liger,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda f: collate(f, tok.pad_token_id or 0),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    model.to(torch.bfloat16)          # Trainer would otherwise write fp32 shards
    trainer.save_model(final)
    # save_pretrained leaves text_config.dtype at the fp32 training dtype even
    # after the cast above; vLLM would then load the bf16 shards upcast
    cfg_path = os.path.join(final, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["dtype"] = cfg["torch_dtype"] = "bfloat16"
    for k in ("text_config", "vision_config"):
        if k in cfg:
            cfg[k]["dtype"] = cfg[k]["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    tok.save_pretrained(final)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, f))
    # the session's frozen decode config: evaluate.py sends no temperature, so
    # vLLM reads it from here (vllm/config/model.py get_diff_sampling_param)
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(
            {"bos_token_id": 2, "cache_implementation": "hybrid", "do_sample": False,
             "temperature": 0.0, "eos_token_id": [1, 106], "pad_token_id": 0,
             "transformers_version": "4.50.0.dev0"}, f, indent=2)
    print(f"[done] saved {final}")


if __name__ == "__main__":
    main()
