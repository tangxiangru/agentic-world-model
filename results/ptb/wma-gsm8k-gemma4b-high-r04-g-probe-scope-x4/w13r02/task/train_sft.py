#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style CoT data.

Rows are rendered with templates/gemma3.jinja (the grader's own template) and
loss is taken on the completion + <end_of_turn> only.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from render import END, build_fewshot_shots, render_prompt, render_target, template_sha256

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)


class SFTRows(Dataset):
    def __init__(self, path: str, tok, max_len: int, system_prob: float = 0.0, seed: int = 0):
        self.rows = []
        n_trunc = 0
        import random

        rng = random.Random(seed)
        shots = build_fewshot_shots() if system_prob > 0 else []
        n_with_sys = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                sys_msg = None
                if shots and rng.random() < system_prob:
                    # half the time the exact 10-shot message the grader builds,
                    # half the time a random subset so the habit is not tied to one prefix
                    if rng.random() < 0.5:
                        chosen = list(range(len(shots)))
                    else:
                        k = rng.randint(1, len(shots) - 1)
                        chosen = sorted(rng.sample(range(len(shots)), k))
                    sys_msg = "\n\n".join(shots[i] for i in chosen)
                    n_with_sys += 1
                p_ids = tok(render_prompt(r["problem"], sys_msg), add_special_tokens=False)["input_ids"]
                c_ids = tok(render_target(r["completion"]), add_special_tokens=False)["input_ids"]
                ids = p_ids + c_ids
                if len(ids) > max_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p_ids) + c_ids
                self.rows.append({"input_ids": ids, "labels": labels, "length": len(ids)})
        self.n_dropped = n_trunc
        self.n_with_sys = n_with_sys

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        n = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            k = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system-prob", type=float, default=0.0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    set_seed(args.seed)
    print("grader template sha256:", template_sha256(), flush=True)

    tok = AutoTokenizer.from_pretrained(args.parent)
    assert tok.convert_tokens_to_ids(END) == 106, tok.convert_tokens_to_ids(END)

    ds = SFTRows(args.data, tok, args.max_len, args.system_prob, args.seed)
    lens = [r["length"] for r in ds.rows]
    lens_sorted = sorted(lens)
    print(
        f"rows kept {len(ds)} (dropped {ds.n_dropped} over max_len={args.max_len}); "
        f"tokens p50={lens_sorted[len(lens)//2]} p99={lens_sorted[int(len(lens)*0.99)]} "
        f"max={lens_sorted[-1]} total={sum(lens)/1e6:.1f}M; rows with a few-shot prefix: {ds.n_with_sys}",
        flush=True,
    )
    # every row must end on the stop token the grader stops at
    bad = sum(1 for r in ds.rows if r["input_ids"][-1] != 106)
    assert bad == 0, f"{bad} rows do not end with <end_of_turn>"

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_train/1e9:.2f}B, frozen {n_frozen/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the processor so a vLLM load of the multimodal config finds everything
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(args.parent).save_pretrained(final)
    except Exception as e:  # noqa: BLE001
        print("processor save skipped:", e)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
