#!/usr/bin/env python3
"""Supervised fine-tuning of google/gemma-3-4b-pt for the inspect_evals/gsm8k grader.

Rows are pre-rendered with render.py so the training string is byte-identical to
what vLLM builds from templates/gemma3.jinja at grading time.  Tokenisation is
done here (not in TRL) so that:
  * no extra BOS is inserted (the template already carries one),
  * every target ends with <end_of_turn>, the terminator the grader stops on,
  * rows longer than --max-seq-len are DROPPED, never truncated.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

STOP = "<end_of_turn>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="jsonl with prompt/completion")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=3072)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--liger", type=int, default=1)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--token-budget", type=int, default=0,
                   help="if >0, batch by padded token count instead of --batch-size")
    return p.parse_args()


class TokenBudgetBatches:
    """Length-sorted batches capped by padded token count.

    Fixed-size batches OOM on this model: gemma-3's 262k vocab plus a 6x spread
    in row length (139-token zero-shot prompts vs 2183-token 10-shot ones) means
    a batch of 32 is harmless for short rows and 93k tokens for long ones.
    Capping tokens/batch instead makes peak activation memory flat.
    """

    def __init__(self, lengths, budget: int, seed: int, epoch: int = 0):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = []
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
        self.epoch = epoch

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        random.Random(self.seed + self.epoch).shuffle(order)
        for b in order:
            yield self.batches[b]


class PackedRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def build_rows(path: str, tok, max_len: int, max_rows: int, seed: int):
    kept, dropped = [], 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            p_ids = tok(r["prompt"], add_special_tokens=False).input_ids
            c_ids = tok(r["completion"], add_special_tokens=False).input_ids
            if len(p_ids) + len(c_ids) > max_len:
                dropped += 1
                continue
            kept.append(
                {
                    "input_ids": p_ids + c_ids,
                    "labels": [-100] * len(p_ids) + c_ids,
                    "length": len(p_ids) + len(c_ids),
                }
            )
    random.Random(seed).shuffle(kept)
    if max_rows:
        kept = kept[:max_rows]
    total = len(kept) + dropped
    print(
        f"[data] kept {len(kept)} rows, dropped {dropped} over max_seq_len "
        f"({dropped / max(total, 1):.4%}); tokens={sum(r['length'] for r in kept):,}",
        flush=True,
    )
    assert dropped / max(total, 1) < 0.02, "more than 2% of rows truncate; raise max_seq_len"
    return kept


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    stop_id = tok.convert_tokens_to_ids(STOP)
    assert stop_id is not None and stop_id > 0, STOP

    rows = build_rows(args.data, tok, args.max_seq_len, args.max_rows, args.seed)
    bad = [r for r in rows[:2000] if r["input_ids"][-1] != stop_id]
    assert not bad, f"{len(bad)} of first 2000 rows do not end with {STOP}"

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    cfg = AutoConfig.from_pretrained(args.model)
    is_mm = cfg.architectures and "ConditionalGeneration" in cfg.architectures[0]
    if is_mm:
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model, dtype=dtype, attn_implementation=args.attn
        )
        frozen = 0
        for name, prm in model.named_parameters():
            if name.startswith("model.vision_tower") or name.startswith(
                "model.multi_modal_projector"
            ):
                prm.requires_grad_(False)
                frozen += prm.numel()
        print(f"[model] froze {frozen/1e6:.1f}M vision params", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation=args.attn
        )
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {trainable/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        lr_scheduler_kwargs=(
            {"min_lr_rate": args.min_lr_ratio} if args.scheduler == "cosine_with_min_lr" else {}
        ),
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        use_liger_kernel=bool(args.liger),
        optim=args.optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        report_to=[],
        seed=args.seed,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )

    ds = PackedRows(rows)
    coll = lambda b: collate(b, tok.pad_token_id)  # noqa: E731

    if args.token_budget:
        from torch.utils.data import DataLoader

        sampler = TokenBudgetBatches(
            [r["length"] for r in rows], args.token_budget, args.seed
        )
        sizes = [len(b) for b in sampler.batches]
        print(
            f"[batches] {len(sampler)} micro-batches; size min/median/max = "
            f"{min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}",
            flush=True,
        )
        targs.group_by_length = False

        class BudgetTrainer(Trainer):
            def get_train_dataloader(self):
                sampler.set_epoch(int(self.state.epoch or 0))
                return self.accelerator.prepare(
                    DataLoader(
                        ds,
                        batch_sampler=sampler,
                        collate_fn=coll,
                        num_workers=targs.dataloader_num_workers,
                        pin_memory=True,
                    )
                )

        trainer = BudgetTrainer(model=model, args=targs, train_dataset=ds, data_collator=coll)
    else:
        trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=coll)
    trainer.train()

    final = os.path.join(args.output_dir, "final")
    model.config.use_cache = True
    # the grader loads final_model/ with vLLM at --gpu-memory-utilization 0.3 (~24 GB);
    # fp32 weights would leave almost no KV cache, so always ship bf16.
    model = model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    # keep the generation config the grader's vLLM reads (stops on <eos> and <end_of_turn>)
    for fname in ("generation_config.json", "preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, fname)
        if os.path.exists(src):
            with open(src) as f:
                blob = f.read()
            with open(os.path.join(final, fname), "w") as f:
                f.write(blob)
    print(f"[done] saved {final}", flush=True)


if __name__ == "__main__":
    main()
