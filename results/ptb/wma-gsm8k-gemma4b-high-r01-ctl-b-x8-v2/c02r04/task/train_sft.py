#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on the pre-tokenized GSM8K-style corpus.

Reads the .npz written by prepare_tokens.py (already rendered through the
grader's own templates/gemma3.jinja and already length-checked), masks the
prompt so loss is completion-only, and trains the language model only (the
vision tower is frozen; it is never in the forward pass for text rows).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
EXTRA_FILES = [
    "preprocessor_config.json", "processor_config.json",
    "added_tokens.json", "special_tokens_map.json", "tokenizer.model",
]


class TokenBatchSampler(torch.utils.data.Sampler):
    """Batches of roughly equal TOKEN count, not equal row count.

    Row lengths run 60..2560 tokens. A fixed row-count batch sized for the
    median row OOMs on a batch of 2560-token rows, and a batch sized for the
    long tail wastes most of the GPU on the median. This shuffles, cuts
    length-sorted megabatches, and fills each batch up to `budget` tokens.
    """

    def __init__(self, lengths, budget: int, max_rows: int, seed: int):
        self.lengths = list(lengths)
        self.budget = budget
        self.max_rows = max_rows
        self.seed = seed
        self.epoch = 0
        self._batches = self._build(0)

    def _build(self, epoch: int):
        g = np.random.default_rng(self.seed + epoch)
        order = g.permutation(len(self.lengths))
        mega = max(self.max_rows * 32, 512)
        batches = []
        for s in range(0, len(order), mega):
            chunk = sorted(order[s:s + mega], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                nmax = max(cur_max, self.lengths[i])
                if cur and (nmax * (len(cur) + 1) > self.budget or len(cur) >= self.max_rows):
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur.append(i)
                    cur_max = nmax
            if cur:
                batches.append(cur)
        g.shuffle(batches)
        return [[int(i) for i in b] for b in batches]

    def set_epoch(self, epoch: int):
        if epoch != self.epoch:
            self.epoch = epoch
            self._batches = self._build(epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class ChunkedCETrainer(Trainer):
    """Loss without ever materialising a [B, T, 262144] logit tensor.

    gemma-3's vocabulary is 262k; at batch 8 x 2560 the fp32 logits alone are
    16.7 GiB and the run OOMs before step 1. Here the LM head + cross-entropy
    run in checkpointed chunks of CHUNK tokens, so peak logit memory is
    CHUNK x 262144 x 4 bytes and the chunks are recomputed in backward.
    """

    CHUNK = 2048
    batch_sampler = None

    def get_train_dataloader(self):
        return self.accelerator.prepare(
            torch.utils.data.DataLoader(
                self.train_dataset,
                batch_sampler=self.batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        base = model.model if hasattr(model, "model") else model.language_model
        out = base(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        hidden = out.last_hidden_state[:, :-1, :]
        tgt = labels[:, 1:]
        keep = tgt.reshape(-1) != -100
        h = hidden.reshape(-1, hidden.size(-1))[keep]
        t = tgt.reshape(-1)[keep]
        lm_head = model.lm_head

        def piece(hc, tc):
            return torch.nn.functional.cross_entropy(
                lm_head(hc).float(), tc, reduction="sum"
            )

        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, h.size(0), self.CHUNK):
            total = total + torch.utils.checkpoint.checkpoint(
                piece, h[i:i + self.CHUNK], t[i:i + self.CHUNK], use_reentrant=False
            )
        denom = num_items_in_batch if num_items_in_batch is not None else h.size(0)
        loss = total / denom
        return (loss, out) if return_outputs else loss


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = f["input_ids"]
            pad = n - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            lab = list(ids) + [-100] * pad
            for i in range(min(f["prompt_len"], len(ids))):
                lab[i] = -100
            labels.append(lab)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_dataset(npz_path: str, limit: int | None):
    z = np.load(npz_path)
    flat, offs, plens = z["flat"], z["offsets"], z["prompt_lens"]
    n = len(plens) if limit is None else min(limit, len(plens))
    ids = [flat[offs[i]:offs[i + 1]].tolist() for i in range(n)]
    return Dataset.from_dict({
        "input_ids": ids,
        "prompt_len": plens[:n].tolist(),
        "length": [len(x) for x in ids],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/sft_v1_tokens.npz")
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--out", default="ckpts/exp-02")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=32, help="max rows per batch")
    ap.add_argument("--token-budget", type=int, default=14000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=1000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    with open(TEMPLATE) as f:
        tok.chat_template = f.read()

    ds = build_dataset(args.npz, args.limit)
    print(f"dataset rows={len(ds)} tokens={sum(ds['length'])/1e6:.1f}M", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.parent,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(type(model).__name__, flush=True)
    for name in ("vision_tower", "multi_modal_projector"):
        mod = getattr(model, name, None) or getattr(getattr(model, "model", None), name, None)
        if mod is not None:
            mod.requires_grad_(False)
            print(f"froze {name}", flush=True)
    model.config.use_cache = False
    # A parent saved by this script carries a greedy generation_config
    # (temperature 0.0 + do_sample False). HF's GenerationConfig.save_pretrained
    # rejects that combination, which killed exp-05's first run AFTER the full
    # epoch had trained. Replace it with a valid one; the greedy JSON is written
    # directly to disk further down, after save_model has run.
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid", do_sample=False,
    )

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        save_safetensors=True,
    )

    trainer = ChunkedCETrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.batch_sampler = TokenBatchSampler(
        ds["length"], budget=args.token_budget, max_rows=args.bs, seed=args.seed
    )
    print(f"batches/epoch={len(trainer.batch_sampler)} "
          f"avg rows/batch={len(ds)/len(trainer.batch_sampler):.1f}", flush=True)
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    for fn in EXTRA_FILES:
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    # greedy decoding, adopted in exp-03: evaluate.py sends no temperature, so
    # vLLM's default sampling params come from this file
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump({
            "bos_token_id": 2,
            "eos_token_id": [1, 106],
            "pad_token_id": 0,
            "cache_implementation": "hybrid",
            "do_sample": False,
            "temperature": 0.0,
            "transformers_version": "4.57.3",
        }, f, indent=2)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
