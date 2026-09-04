#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on the grader's own prompt format.

Rows come from data/*.jsonl as {prompt, target}; `prompt` is byte-exact what
vLLM is fed at eval time and `target` ends in <end_of_turn> (token 106, which
is in the base generation_config's eos_token_id list).  Loss is masked to the
target tokens only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

IGNORE = -100


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, report=True):
        self.rows = []
        n_trunc = 0
        lens = []
        with open(path) as f:
            raw = [json.loads(l) for l in f]
        prompts = [r["prompt"] for r in raw]
        targets = [r["target"] for r in raw]
        p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
        t_ids = tok(targets, add_special_tokens=False)["input_ids"]
        for p, t in zip(p_ids, t_ids):
            ids = p + t
            lens.append(len(ids))
            if len(ids) > max_seq_len:
                n_trunc += 1
                continue  # dropping beats truncating: a truncated row has no stop token
            labels = [IGNORE] * len(p) + list(t)
            self.rows.append((ids, labels))
        if report:
            lens = np.array(lens)
            print(
                f"[data] {path}: n={len(lens)} p50={int(np.percentile(lens,50))} "
                f"p95={int(np.percentile(lens,95))} p99={int(np.percentile(lens,99))} "
                f"max={int(lens.max())} dropped>{max_seq_len}={n_trunc} "
                f"({n_trunc/len(lens):.3%}) kept={len(self.rows)} "
                f"target_tokens={int(sum(len(l)-l.count(IGNORE) for _,l in self.rows[:1000])/min(1000,len(self.rows)))}avg",
                flush=True,
            )
            assert n_trunc / len(lens) < 0.02, "more than 2% of rows exceed max_seq_len"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [IGNORE] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Length-bucketed batches with a fixed padded-token budget.

    Rows here run from ~120 to ~3000 tokens (p50 329, p99 1959), so a fixed
    per-device batch size either wastes the GPU on the short rows or OOMs on the
    long ones. Batching to a token budget keeps every micro-batch the same size
    in tokens, which is what actually determines both throughput and peak memory.
    """

    def __init__(self, lengths, max_tokens, max_bs, seed, megabatch=4096):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.max_bs = max_bs
        self.seed = seed
        self.megabatch = megabatch
        self.epoch = 0
        self._batches = self._build(0)

    def _build(self, epoch):
        rng = random.Random(self.seed + epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.megabatch):
            chunk = sorted(idx[s : s + self.megabatch], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                m = max(cur_max, self.lengths[i])
                if cur and (m * (len(cur) + 1) > self.max_tokens or len(cur) >= self.max_bs):
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur.append(i)
                    cur_max = m
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self.epoch = epoch
            self._batches = self._build(epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class SparseLossTrainer(Trainer):
    """Cross-entropy over the labelled positions only.

    Gemma-3's vocabulary is 262144, so materialising logits for every position of
    a 4x3072 micro-batch costs ~25 GB and OOMs an 80 GB H100. Under completion-only
    masking only ~25-35% of positions carry a label, so the lm_head is applied to
    just those hidden states. This is exactly the same loss, not an approximation.
    """

    batch_sampler = None

    def get_train_dataloader(self):
        if self.batch_sampler is None:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        core = model.module if hasattr(model, "module") else model
        out = core.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        hidden = out.last_hidden_state              # (B, T, H)
        shift_h = hidden[:, :-1, :]
        shift_y = labels[:, 1:]
        sel = shift_y != IGNORE
        h_sel = shift_h[sel]                        # (N, H)
        y_sel = shift_y[sel]                        # (N,)
        logits = core.lm_head(h_sel).float()
        loss = torch.nn.functional.cross_entropy(logits, y_sel, reduction="sum")
        denom = num_items_in_batch if num_items_in_batch is not None else y_sel.numel()
        loss = loss / denom
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max-tokens-per-batch", type=int, default=16384)
    ap.add_argument("--max-bs", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.convert_tokens_to_ids(fmt.STOP_TOKEN) == 106

    ds = SFTRows(args.data, tok, args.max_seq_len)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    # text-only corpus: the SigLIP tower and the projector get no gradient signal
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    # A parent produced by this script ships {do_sample: false, temperature: 0.0} so
    # that vLLM decodes greedily. transformers loads that into model.generation_config
    # and then REFUSES to save it (GenerationConfig.validate: temperature is a
    # sample-only field), which kills the first checkpoint save mid-run. Neutralise it
    # here; the greedy config is written back by hand after training.
    for f in ("temperature", "top_k", "top_p", "min_p"):
        if getattr(model.generation_config, f, None) is not None:
            setattr(model.generation_config, f, None)
    model.generation_config.do_sample = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params {n_train/1e9:.3f}B of {sum(p.numel() for p in model.parameters())/1e9:.3f}B", flush=True)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=6,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        group_by_length=False,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        save_safetensors=True,
    )

    trainer = SparseLossTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    if args.max_tokens_per_batch > 0:
        trainer.batch_sampler = TokenBudgetBatches(
            [len(r[0]) for r in ds.rows], args.max_tokens_per_batch, args.max_bs, args.seed
        )
        print(f"[data] {len(trainer.batch_sampler)} micro-batches/epoch at <= {args.max_tokens_per_batch} padded tokens", flush=True)
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the grader's decode deterministic: the base generation_config ships
    # do_sample=true/top_k=64/top_p=0.95 and vLLM reads it.
    gc_path = os.path.join(final, "generation_config.json")
    # vLLM reads only repetition_penalty/temperature/top_k/top_p/min_p/max_new_tokens
    # out of this file (ModelConfig.get_diff_sampling_param); do_sample is ignored.
    # temperature 0.0 is therefore the only way to make the grader decode greedily.
    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": False,
        "temperature": 0.0,
    }
    with open(gc_path, "w") as f:
        json.dump(gc, f, indent=2)
    # vLLM builds an AutoProcessor for Gemma3ForConditionalGeneration; those files
    # are not written by save_model/save_pretrained on the tokenizer.
    import shutil
    for extra in ("preprocessor_config.json", "processor_config.json"):
        src_extra = os.path.join(args.model, extra)
        if os.path.exists(src_extra):
            shutil.copyfile(src_extra, os.path.join(final, extra))
    print(f"[done] saved {final}", flush=True)


if __name__ == "__main__":
    main()
