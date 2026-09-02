#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on grader-formatted math CoT.

Prompt/target are rendered with scripts/fmt.py, which is verified byte-for-byte
against templates/gemma3.jinja (the template the grader passes to vLLM).
Loss is on the completion only; every target ends with <end_of_turn>.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402


class SelectiveLossTrainer(Trainer):
    """Cross-entropy over completion positions only.

    Gemma-3's vocab is 262144, so materialising logits for every position of a
    3456-token row OOMs an 80 GB H100 (18 GiB for one CE call). Under
    completion-only loss all but ~350 positions per row are ignore_index
    anyway, so we run lm_head on the labelled positions alone. The value is
    identical to the stock forward's CrossEntropyLoss(ignore_index=-100) mean.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        base = model.module.model if hasattr(model, "module") else model.model
        head = model.module.lm_head if hasattr(model, "module") else model.lm_head
        out = base(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        hs = out[0][:, :-1, :]
        lb = labels[:, 1:]
        mask = lb != -100
        sel_h = hs[mask]
        sel_l = lb[mask]
        n = sel_l.numel()
        total = None
        for i in range(0, n, LOSS_CHUNK):
            lg = head(sel_h[i : i + LOSS_CHUNK]).float()
            part = torch.nn.functional.cross_entropy(
                lg, sel_l[i : i + LOSS_CHUNK], reduction="sum"
            )
            total = part if total is None else total + part
        loss = total / n
        return (loss, out) if return_outputs else loss


LOSS_CHUNK = 4096


class Rows(Dataset):
    def __init__(self, examples):
        self.ex = examples

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def build(tok, rows, fewshot_frac, max_len, seed, system_msg):
    rng = random.Random(seed)
    out, dropped, fs_n = [], 0, 0
    for r in rows:
        use_fs = rng.random() < fewshot_frac
        prompt = fmt.render_prompt(r["question"], system_msg if use_fs else None)
        target = fmt.render_target(r["completion"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > max_len:
            dropped += 1
            continue
        fs_n += int(use_fs)
        out.append(
            {
                "input_ids": p_ids + t_ids,
                "labels": [-100] * len(p_ids) + t_ids,
                "length": len(p_ids) + len(t_ids),
            }
        )
    print(f"built {len(out)} rows, dropped {dropped} over max_len, {fs_n} with fewshot prefix")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=fmt.SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(fmt.SNAPSHOT)
    system_msg = open(os.path.join(fmt.TASK_DIR, "data", "fewshot_system.txt")).read()

    rows = [json.loads(l) for l in open(args.data)]
    if args.n > 0:
        rows = rows[: args.n]
    ex = build(tok, rows, args.fewshot_frac, args.max_seq_len, args.seed, system_msg)
    ntok = sum(e["length"] for e in ex)
    print(f"total tokens {ntok/1e6:.1f}M, mean {ntok/len(ex):.0f}, max {max(e['length'] for e in ex)}")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {frozen/1e6:.0f}M, trainable {trainable/1e6:.0f}M")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        group_by_length=True,
        length_column_name="length",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = SelectiveLossTrainer(
        model=model,
        args=targs,
        train_dataset=Rows(ex),
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()
    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the processor files so the saved dir mirrors the base snapshot layout
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(fmt.SNAPSHOT, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, f)):
            import shutil

            shutil.copy(src, os.path.join(final, f))
    print("saved", final)


if __name__ == "__main__":
    main()
