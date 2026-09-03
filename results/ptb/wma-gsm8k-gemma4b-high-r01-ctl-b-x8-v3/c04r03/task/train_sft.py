#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered {prompt, completion} jsonl.

The jsonl is produced by build_data.py, which renders the prompt with the
grader's own templates/gemma3.jinja and terminates every completion with
<end_of_turn> -- the token vLLM stops on. This script therefore only has to
tokenize with add_special_tokens=False (the template already emits <bos>,
and vLLM's chat endpoint defaults add_special_tokens=False too).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")


class PackedRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


@dataclass
class Collator:
    pad_id: int

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Length-sorted batches capped at a fixed number of padded tokens.

    Rows here run from ~120 to ~2500 tokens (the few-shot-prefixed ones are the
    long tail). A fixed sequence count would either OOM on the long rows or
    waste most of the H100 on the short ones, so batch by token budget instead.
    """

    def __init__(self, lengths, budget, seed=0, bucket=2048):
        self.batches = []
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        cur, cur_max = [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and m * (len(cur) + 1) > budget:
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur.append(i)
                cur_max = m
        if cur:
            self.batches.append(cur)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        import random
        r = random.Random(self.seed + self.epoch)
        order = list(range(len(self.batches)))
        r.shuffle(order)
        self.epoch += 1
        for i in order:
            yield self.batches[i]


def tokenize_all(path, tok, max_seq_len, max_rows, min_completion_tokens=8):
    rows, n_trunc, n_total = [], 0, 0
    with open(path) as fh:
        for line in fh:
            if max_rows and n_total >= max_rows:
                break
            n_total += 1
            r = json.loads(line)
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"]
            if len(c) < min_completion_tokens:
                continue
            if len(p) + len(c) > max_seq_len:
                n_trunc += 1
                continue          # drop, never truncate: a truncated target
                                  # silently removes the stop token (pitfall
                                  # seq_len_truncation / eos_mismatch)
            rows.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    return rows, n_total, n_trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=14336,
                    help="max padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--no-liger", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    rows, n_total, n_trunc = tokenize_all(args.data, tok, args.max_seq_len, args.max_rows)
    lens = sorted(len(r["input_ids"]) for r in rows)
    print(f"rows kept {len(rows)} / read {n_total}; dropped over-length {n_trunc} "
          f"({n_trunc / max(n_total,1):.3%})", flush=True)
    print(f"len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}",
          flush=True)
    if n_trunc / max(n_total, 1) > 0.02:
        print("WARNING: more than 2% of rows dropped for length", flush=True)

    if not args.no_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger gemma3 kernels applied", flush=True)

    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn, config=cfg,
    )
    print("model class:", type(model).__name__, flush=True)
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e9:.2f}B", flush=True)
    model.config.use_cache = False
    # A parent checkpoint saved by this script carries the greedy decode config
    # (do_sample False + temperature 0.0 + top_k -1) that vLLM needs. Transformers
    # refuses to SAVE that combination -- GenerationConfig.validate() raises and
    # every Trainer checkpoint write dies. Hold a valid config during training and
    # write the greedy file back by hand at the end.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid",
    )

    sampler = TokenBudgetBatches([len(r["input_ids"]) for r in rows],
                                 args.token_budget, seed=args.seed)
    print(f"micro-batches/epoch: {len(sampler)}  "
          f"avg rows/batch: {len(rows)/len(sampler):.1f}", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        max_grad_norm=1.0,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        save_safetensors=True,
        accelerator_config={"split_batches": False, "dispatch_batches": False},
    )

    collator = Collator(pad_id)
    dataset = PackedRows(rows)

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self):
            return self.accelerator.prepare(DataLoader(
                dataset, batch_sampler=sampler, collate_fn=collator,
                num_workers=2, pin_memory=True,
            ))

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    final = Path(args.out) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    # the grader loads this directory with vLLM: carry over every non-weight
    # asset from the base snapshot (processor/preprocessor config, etc.)
    for f in os.listdir(BASE):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        dst = final / f
        if not dst.exists():
            shutil.copy(os.path.join(BASE, f), dst)
    # greedy decoding: inspect_ai sends no temperature, so vLLM falls back to
    # this file's defaults; the base file has none -> temperature 1.0 sampling.
    (final / "generation_config.json").write_text(json.dumps({
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
    }, indent=2))
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
