"""Completion-only, sequence-packed SFT of google/gemma-3-4b-pt.

Rows come from scripts/build_data.py already rendered with the grader's exact
chat template (scripts/fmt.py, verified byte-for-byte by verify_template.py).
This script only tokenizes, masks the prompt, packs, and trains.

Packing: examples are concatenated into flat blocks of --pack-len with
`position_ids` restarting at 0 on every example boundary and batch_size 1.
transformers' `_is_packed_sequence` detects exactly that shape and routes
flash-attention-2 through `flash_attn_varlen_func`, so there is no
cross-example attention. Trailing padding in a block is its own segment with
labels -100.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
PAD_ID = 0


# ---------------------------------------------------------------- data

class PackedBlocks(TorchDataset):
    """Fixed-length packed blocks produced by scripts/pack_data.py."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.ids, self.lab, self.pos = z["ids"], z["lab"], z["pos"]
        print(f"[data] blocks={len(self.ids)} pack_len={self.ids.shape[1]} "
              f"loss_tokens={(self.lab != -100).sum()/1e6:.1f}M", flush=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.lab[i], "position_ids": self.pos[i]}


class Collator:
    def __call__(self, feats):
        return {
            "input_ids": torch.from_numpy(np.stack([f["input_ids"] for f in feats])).long(),
            "labels": torch.from_numpy(np.stack([f["labels"] for f in feats])).long(),
            "position_ids": torch.from_numpy(np.stack([f["position_ids"] for f in feats])).long(),
        }


# ---------------------------------------------------------------- loss

def _ce_chunk(logits_chunk, labels_chunk):
    return F.cross_entropy(logits_chunk.float(), labels_chunk, ignore_index=-100, reduction="sum")


class PackedTrainer(Trainer):
    ce_chunk = 2048

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        out = model(**inputs)
        logits = out.logits
        V = logits.size(-1)
        sl = logits[:, :-1, :].reshape(-1, V)
        sy = labels[:, 1:].reshape(-1)
        ntok = (sy != -100).sum()
        total = sl.new_zeros((), dtype=torch.float32)
        for s in range(0, sy.numel(), self.ce_chunk):
            lc = sl[s:s + self.ce_chunk]
            yc = sy[s:s + self.ce_chunk]
            if self.model.training and torch.is_grad_enabled():
                total = total + torch.utils.checkpoint.checkpoint(
                    _ce_chunk, lc, yc, use_reentrant=False)
            else:
                total = total + _ce_chunk(lc, yc)
        loss = total / ntok.clamp(min=1)
        return (loss, out) if return_outputs else loss


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True, help="npz from scripts/pack_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=12)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    ds = PackedBlocks(args.packed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    print("[model]", type(model).__name__, flush=True)
    model.config.use_cache = False
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    print(f"[model] trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.0f}M",
          flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.log_steps,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=2,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        optim=args.optim,
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        save_safetensors=True,
    )

    trainer = PackedTrainer(model=model, args=targs, train_dataset=ds, data_collator=Collator())
    # our compute_loss returns a per-token mean, so Trainer must do the /accum itself
    trainer.model_accepts_loss_kwargs = False
    trainer.train()

    final = os.path.join(args.out, "final")
    # transformers refuses to serialise do_sample=False together with
    # temperature=0.0, and a greedy parent checkpoint carries exactly that pair.
    # Save a valid config, then write the greedy values back onto the json below.
    gcfg = model.generation_config
    gcfg.do_sample = False
    gcfg.temperature = None
    gcfg.top_k = None
    gcfg.top_p = None
    trainer.save_model(final)
    tok.save_pretrained(final)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    # exp-03: vLLM takes its default SamplingParams from generation_config.json and
    # the harness never overrides them, so greedy has to be written into the artefact.
    gc_path = os.path.join(final, "generation_config.json")
    gc = json.load(open(gc_path))
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("[done] saved", final, "generation_config:", gc, flush=True)


if __name__ == "__main__":
    main()
