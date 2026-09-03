"""Completion-only SFT for google/gemma-3-4b-pt, throughput-tuned.

Same contract as train_sft.py (grader's template, prompt masked, target ends
with <end_of_turn>) with three changes that only affect speed:
  * rows are tokenised once up front, in batch, and cached as int32 arrays
  * batches are length-grouped, so padding waste collapses on a corpus whose
    rows run 150-900 tokens
  * rows longer than max_seq_len are dropped rather than head-truncated, so no
    row is ever trained with a mangled question
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_utils as P  # noqa: E402

IGNORE = -100


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, fewshot_frac=0.0, seed=0, limit=None):
        rows = [json.loads(l) for l in open(path)]
        if limit:
            rows = rows[:limit]
        rng = random.Random(seed)
        flags = [rng.random() < fewshot_frac for _ in rows]
        self.eot_id = tok.convert_tokens_to_ids("<end_of_turn>")

        prompts = [P.eval_prompt(tok, r["question"], fewshot=f) for r, f in zip(rows, flags)]
        targets = [r["completion"] for r in rows]
        p_enc = tok(prompts, add_special_tokens=False)["input_ids"]
        t_enc = tok(targets, add_special_tokens=False)["input_ids"]

        self.ids, self.n_prompt, self.dropped = [], [], 0
        for p, t in zip(p_enc, t_enc):
            if t[-1] != self.eot_id:
                self.dropped += 1
                continue
            if len(p) + len(t) > max_seq_len:
                self.dropped += 1
                continue
            self.ids.append(np.asarray(p + t, dtype=np.int32))
            self.n_prompt.append(len(p))
        self.lengths = np.asarray([len(x) for x in self.ids])
        self.n_input = len(rows)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ids = self.ids[i].tolist()
        labels = [IGNORE] * self.n_prompt[i] + ids[self.n_prompt[i]:]
        return {"input_ids": ids, "labels": labels}


class LengthBucketSampler(Sampler):
    """Shuffle, then sort within megabatches so each batch is near-uniform length."""

    def __init__(self, lengths, batch_size, mega=64, seed=0, epoch=0):
        self.lengths, self.bs, self.mega, self.seed, self.epoch = lengths, batch_size, mega, seed, epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        idx = g.permutation(len(self.lengths))
        chunk = self.bs * self.mega
        out = []
        for i in range(0, len(idx), chunk):
            block = idx[i:i + chunk]
            out.extend(block[np.argsort(self.lengths[block], kind="stable")])
        # shuffle the order of the batches themselves so length is not a schedule
        batches = [out[i:i + self.bs] for i in range(0, len(out), self.bs)]
        g.shuffle(batches)
        return iter([i for b in batches for i in b])


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [IGNORE] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.fewshot_frac, args.seed, args.limit)
    L = np.sort(ds.lengths)
    print(f"[preflight] {len(ds)}/{ds.n_input} rows kept ({ds.dropped} dropped: over "
          f"max_seq_len={args.max_seq_len} or missing stop token = "
          f"{ds.dropped/max(1,ds.n_input):.3%}); token len p50={L[len(L)//2]} "
          f"p99={L[int(len(L)*0.99)]} max={L[-1]}")
    ex = ds[0]
    print("[preflight] loss region of row 0 (tail):")
    print(repr(tok.decode([t for t in ex["labels"] if t != IGNORE])[-300:]))
    assert ds.dropped / max(1, ds.n_input) < 0.02, "too many rows dropped; raise --max-seq-len"
    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"[model] froze {n_frozen/1e6:.0f}M vision params; trainable "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        logging_steps=20,
        bf16=True,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=3,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        accelerator_config={"dispatch_batches": False},
    )

    class T(Trainer):
        def _get_train_sampler(self, *a, **k):
            return LengthBucketSampler(ds.lengths, args.bs, seed=args.seed)

    trainer = T(model=model, args=targs, train_dataset=ds,
                data_collator=lambda b: collate(b, tok.pad_token_id or 0))
    res = trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # noqa: BLE001
        print("processor save skipped:", e)
    with open(os.path.join(args.out, "train_summary.json"), "w") as f:
        json.dump({"final_loss": res.training_loss, "steps": res.global_step,
                   "kept_rows": len(ds), "args": vars(args)}, f, indent=1)
    print("saved", final, res.training_loss, res.global_step)


if __name__ == "__main__":
    main()
