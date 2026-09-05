#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on a prompt/completion jsonl.

The jsonl carries the already-rendered strings (scripts/build_data.py), so what
the trainer sees is byte-identical to what the grader will send. Tokenisation
here uses add_special_tokens=False because the rendered prompt already starts
with <bos> (the chat template emits it).

Loss is on completion tokens only.
"""

import argparse
import json
import math
import os
import random
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


class PromptCompletionDataset(Dataset):
    def __init__(self, path, tok, max_seq_len, report=True):
        self.rows = []
        n_trunc = 0
        lens = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                ids = p + c
                lens.append(len(ids))
                if len(ids) > max_seq_len:
                    n_trunc += 1
                    continue  # never truncate: a cut target teaches a missing stop token
                labels = [-100] * len(p) + list(c)
                self.rows.append((ids, labels))
        if report:
            lens.sort()
            print(
                f"[data] {path}: kept {len(self.rows)}, dropped_over_len {n_trunc} "
                f"({n_trunc / max(1, len(lens)):.3%}), "
                f"p50 {lens[len(lens) // 2]} p95 {lens[int(0.95 * len(lens))]} max {lens[-1]}",
                flush=True,
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
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
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--model", default=fmt.BASE_SNAPSHOT)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--liger", type=int, default=1)
    args = ap.parse_args()

    if args.liger:
        # fused linear cross-entropy: the 262k-vocab logit tensor is never
        # materialised, which is what OOMed the first smoke run at bs=8
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("[liger] gemma3 kernels applied", flush=True)

    set_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(fmt.BASE_SNAPSHOT)
    tok.chat_template = fmt.chat_template()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    ds = PromptCompletionDataset(args.train_file, tok, args.max_seq_len)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # HF refuses to save a GenerationConfig with do_sample=False and a temperature
    # set (it raises inside save_pretrained and kills the run at the first
    # checkpoint). vLLM does not validate, so the greedy file is written by hand
    # at the end; keep the in-memory one valid so saving works.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = True
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    # text-only task: the vision tower and projector are dead weight
    frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith(
            "model.multi_modal_projector"
        ):
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {trainable/1e9:.2f}B  frozen {frozen/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, pad_id),
    )
    out = trainer.train()
    print("[train]", out.metrics, flush=True)

    final = os.path.join(args.output_dir, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    import shutil

    for f in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(fmt.BASE_SNAPSHOT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, f))
    # vLLM takes its default sampling params from this file (exp-03: greedy is
    # +9.3 points over the base config's do_sample/top_k=64/top_p=0.95).
    # eos_token_id keeps 106 = <end_of_turn>, the terminator the grader's template uses.
    with open(os.path.join(final, "generation_config.json"), "w") as fh:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "cache_implementation": "hybrid", "do_sample": False,
                   "temperature": 0.0}, fh, indent=2)
    with open(os.path.join(final, "train_metrics.json"), "w") as fh:
        json.dump(out.metrics, fh, indent=2)
    print("[saved]", final, flush=True)


if __name__ == "__main__":
    main()
