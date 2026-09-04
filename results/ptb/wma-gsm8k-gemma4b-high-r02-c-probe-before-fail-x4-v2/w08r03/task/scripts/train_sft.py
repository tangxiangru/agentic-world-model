#!/usr/bin/env python3
"""Completion-only SFT of gemma-3-4b-pt on pre-rendered prompt/completion jsonl.

The jsonl rows are already rendered with templates/gemma3.jinja (see scripts/fmt.py),
so this script does no templating: it tokenizes prompt and completion with
add_special_tokens=False (the prompt already carries <bos>), concatenates, and
masks the prompt tokens out of the loss.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

IGNORE = -100


class PromptCompletionDataset(Dataset):
    def __init__(self, path: str, tok, max_seq_len: int, drop_long: bool = True):
        self.rows = []
        n_long = 0
        n_total = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                n_total += 1
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_seq_len:
                    n_long += 1
                    if drop_long:
                        continue
                    p = p[: max_seq_len - len(c)]
                self.rows.append((p, c))
        print(f"[data] {path}: kept {len(self.rows)}/{n_total} rows "
              f"({n_long} over max_seq_len={max_seq_len})", flush=True)
        lens = sorted(len(p) + len(c) for p, c in self.rows)
        print(f"[data] token length p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} "
              f"max={lens[-1]} total={sum(lens)/1e6:.1f}M", flush=True)
        self.lengths = [len(p) + len(c) for p, c in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        labels = [IGNORE] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class TokenBudgetBatches:
    """Length-bucketed micro-batches with a fixed padded-token budget.

    The row length distribution here is bimodal (p50 334, p95 1660) because a
    quarter of the rows carry a k-shot prefix. A fixed per_device_train_batch_size
    therefore either wastes the GPU on the short rows or OOMs on the long ones;
    batching to a token budget keeps every micro-batch about the same size in
    padded tokens, which is what actually drives memory and step time.
    """

    def __init__(self, lengths, budget: int, max_bs: int, seed: int):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = []
        cur, cur_max = [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and (m * (len(cur) + 1) > budget or len(cur) >= max_bs):
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur.append(i)
                cur_max = m
        if cur:
            self.batches.append(cur)
        self.epoch = 0
        self.seed = seed
        sizes = [len(b) for b in self.batches]
        print(f"[batches] {len(self.batches)} micro-batches, size min={min(sizes)} "
              f"median={sorted(sizes)[len(sizes)//2]} max={max(sizes)} "
              f"(budget={budget} padded tokens)", flush=True)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        import random as _r
        rng = _r.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.batches)))
        rng.shuffle(order)
        for j in order:
            yield self.batches[j]


class Collator:
    def __init__(self, pad_id: int):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--token-budget", type=int, default=8192,
                    help="padded tokens per micro-batch; 0 disables token batching")
    ap.add_argument("--max-bs", type=int, default=64)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = PromptCompletionDataset(args.data, tok, args.max_seq_len)

    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn, config=cfg,
    )
    model.config.use_cache = False
    # the vision tower is dead weight for this task: freeze it so no optimizer
    # state is allocated for it, but keep the weights so final_model loads as the
    # same architecture the grader's vLLM expects.
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_train/1e9:.3f}B, frozen {n_frozen/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
    )

    collator = Collator(tok.pad_token_id or 0)

    if args.token_budget:
        from torch.utils.data import DataLoader

        batches = TokenBudgetBatches(ds.lengths, args.token_budget, args.max_bs, args.seed)

        class TokenBudgetTrainer(Trainer):
            def get_train_dataloader(self):
                return self.accelerator.prepare(DataLoader(
                    ds, batch_sampler=batches, collate_fn=collator,
                    num_workers=2, pin_memory=True))

        trainer = TokenBudgetTrainer(model=model, args=targs, train_dataset=ds,
                                     data_collator=collator)
    else:
        trainer = Trainer(model=model, args=targs, train_dataset=ds,
                          data_collator=collator)
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the processor files so vLLM loads the multimodal config cleanly
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, fn)
        if os.path.exists(src):
            with open(src) as fi, open(os.path.join(final, fn), "w") as fo:
                fo.write(fi.read())
    print(f"[peak] {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    print(f"[done] saved {final}", flush=True)


if __name__ == "__main__":
    main()
