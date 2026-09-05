#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style data.

Renders prompts with the SAME string the grader's chat template
(templates/gemma3.jinja) produces, and ends every target with <end_of_turn>,
the terminator vLLM stops on.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
from array import array

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

SNAP = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
STOP = "<end_of_turn>"
TOK_FILES = [
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "preprocessor_config.json", "processor_config.json",
]


def render_prompt(user_content: str) -> str:
    """Exactly what templates/gemma3.jinja emits, minus the leading <bos>."""
    return ("<start_of_turn>user\n" + user_content.strip()
            + "<end_of_turn>\n<start_of_turn>model\n")


class SFTData(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = list(p) + list(c)
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = (n + 7) // 8 * 8
        ids, lab, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            lab.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids),
                "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


class LenTrainer(Trainer):
    lengths: list[int] = []

    def _get_train_sampler(self, *a, **kw):
        if self.args.group_by_length:
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=self.lengths, model_input_name="input_ids")
        return super()._get_train_sampler(*a, **kw)


def load_fewshot_pool(n, seed):
    """k-shot exemplars in the grader's few-shot format, from the gsm8k TRAIN split."""
    import pyarrow.parquet as pq
    f = glob.glob("/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet")
    tbl = pq.read_table(f[0]).to_pylist()
    rng = random.Random(seed)
    rng.shuffle(tbl)
    pool = []
    for r in tbl:
        q = r["question"].strip()
        a = r["answer"].split("####")
        tgt = a.pop().strip()
        reasoning = "####".join(a).strip()
        if len(q) > 700 or len(reasoning) > 900:
            continue
        pool.append(f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {tgt}")
        if len(pool) >= n:
            break
    return pool


def build_rows(path, tok, max_len, fewshot_frac, pool, seed, bos_id):
    rng = random.Random(seed)
    rows, dropped, total = [], 0, 0
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            total += 1
            user = d["prompt"]
            if pool and rng.random() < fewshot_frac:
                k = rng.choice([1, 2, 3])
                user = "\n\n".join(rng.sample(pool, k)) + "\n\n" + user
            p = tok(render_prompt(user), add_special_tokens=False)["input_ids"]
            c = tok(d["answer"], add_special_tokens=False)["input_ids"]
            if 1 + len(p) + len(c) > max_len:
                dropped += 1
                continue
            rows.append((array("i", [bos_id] + p), array("i", c)))
    print(f"rows={len(rows)} dropped_too_long={dropped} ({dropped/max(1,total):.4%})",
          flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    bos_id = tok.convert_tokens_to_ids("<bos>")
    pad_id = tok.convert_tokens_to_ids("<pad>")
    assert tok.convert_tokens_to_ids(STOP) == 106, "stop token id changed"

    pool = load_fewshot_pool(600, args.seed) if args.fewshot_frac > 0 else []
    rows = build_rows(args.data, tok, args.max_seq_len, args.fewshot_frac,
                      pool, args.seed, bos_id)
    if args.limit:
        rows = rows[: args.limit]
    ds = SFTData(rows)
    lengths = [len(p) + len(c) for p, c in rows]
    print(f"tokens={sum(lengths)/1e6:.1f}M  mean_len={sum(lengths)/len(lengths):.0f}",
          flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn)
    root = getattr(model, "model", model)
    frozen = 0
    for attr in ("vision_tower", "multi_modal_projector"):
        for holder in (model, root):
            m = getattr(holder, attr, None)
            if m is not None:
                m.requires_grad_(False)
                frozen += sum(p.numel() for p in m.parameters())
                break
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen={frozen/1e9:.3f}B trainable={trainable/1e9:.3f}B", flush=True)

    # A parent that already carries the greedy generation_config (do_sample=False
    # with temperature=0.0 / top_k=-1) makes GenerationConfig.save_pretrained raise
    # under strict validation, which kills the run at the first checkpoint save.
    # Keep a valid config on the model; the greedy one is written to disk below.
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = True, 1.0, 0.95, 64

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        report_to=[],
        gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        dataloader_num_workers=2,
        seed=args.seed,
        remove_unused_columns=False,
    )
    LenTrainer.lengths = lengths
    trainer = LenTrainer(model=model, args=targs, train_dataset=ds,
                         data_collator=Collator(pad_id))
    trainer.train()

    final = os.path.join(args.out, "final")
    os.makedirs(final, exist_ok=True)
    trainer.model.config.use_cache = True
    trainer.model.save_pretrained(final, safe_serialization=True)
    for f in TOK_FILES:
        src = os.path.join(SNAP, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, f))
    greedy = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
              "cache_implementation": "hybrid", "do_sample": False,
              "temperature": 0.0, "top_p": 1.0, "top_k": -1,
              "transformers_version": "4.57.3"}
    for d in [final] + glob.glob(os.path.join(args.out, "checkpoint-*")):
        with open(os.path.join(d, "generation_config.json"), "w") as fh:
            json.dump(greedy, fh, indent=2)
    print("SAVED", final, flush=True)


if __name__ == "__main__":
    main()
