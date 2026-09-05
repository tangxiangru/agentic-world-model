#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on a GSM8K-style corpus.

Prompts are rendered with the EXACT chat template the grader uses
(templates/gemma3.jinja), completion-only loss, targets terminated by
<end_of_turn> (the token vLLM stops on under that template).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MATH_PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response should be of '
    'the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    "{prompt}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" '
    "(without quotes) where $ANSWER is the answer to the problem, and you do not need to use a "
    "\\boxed command.\n\nReasoning:"
)


def build_examples(rows, tok, fewshot_pool, max_seq_len, rng):
    """Tokenize into {input_ids, labels, length}; prompt tokens are masked."""
    out, dropped = [], 0
    for r in rows:
        user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
        msgs = []
        k = int(r.get("nshot", 0) or 0)
        if k:
            shots = rng.sample(fewshot_pool, k)
            msgs.append({"role": "system", "content": "\n\n".join(shots)})
        msgs.append({"role": "user", "content": user})
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        target_text = r["completion"]

        p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        t_ids = tok(target_text, add_special_tokens=False)["input_ids"]
        ids = p_ids + t_ids
        if len(ids) > max_seq_len:
            dropped += 1
            continue
        labels = [-100] * len(p_ids) + t_ids
        out.append({"input_ids": ids, "labels": labels, "length": len(ids)})
    return out, dropped


class TokenBudgetBatchSampler(Sampler):
    """Length-grouped batches with a cap on padded tokens per micro-batch."""

    def __init__(self, lengths, token_budget, max_bs, seed, megabatch=4096):
        self.lengths = lengths
        self.token_budget = token_budget
        self.max_bs = max_bs
        self.seed = seed
        self.megabatch = megabatch
        self.epoch = 0
        self._batches = self._build(seed)

    def _build(self, seed):
        rng = random.Random(seed)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.megabatch):
            chunk = sorted(idx[s : s + self.megabatch], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in chunk:
                nmax = max(cur_max, self.lengths[i])
                if cur and (nmax * (len(cur) + 1) > self.token_budget or len(cur) >= self.max_bs):
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur, cur_max = cur + [i], nmax
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = self._build(self.seed + epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def collate(features, pad_id):
    n = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        pad = n - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * pad)
        labels.append(f["labels"] + [-100] * pad)
        attn.append([1] * len(f["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


class BatchSamplerTrainer(Trainer):
    def __init__(self, *a, batch_sampler=None, pad_id=0, **kw):
        super().__init__(*a, **kw)
        self._batch_sampler = batch_sampler
        self._pad_id = pad_id

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=lambda f: collate(f, self._pad_id),
            num_workers=2,
            pin_memory=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="/home/ben/task/templates/gemma3.jinja")
    ap.add_argument("--fewshot-pool", default="/home/ben/task/data/fewshot_pool.json")
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--max-bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--grad-ckpt", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open(args.template).read()

    fewshot_pool = json.load(open(args.fewshot_pool))
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]

    examples, dropped = build_examples(rows, tok, fewshot_pool, args.max_seq_len, rng)
    lens = sorted(e["length"] for e in examples)
    print(
        f"examples={len(examples)} dropped_over_max_seq_len={dropped} "
        f"p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]} "
        f"total_tokens={sum(lens)}",
        flush=True,
    )
    assert dropped / max(1, len(rows)) < 0.02, "more than 2% of rows truncate"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen vision params: {n_frozen/1e6:.1f}M", flush=True)

    sampler = TokenBudgetBatchSampler(
        [e["length"] for e in examples], args.token_budget, args.max_bs, args.seed
    )
    steps_per_epoch = math.ceil(len(sampler) / args.grad_accum)
    print(f"micro_batches={len(sampler)} optimizer_steps/epoch={steps_per_epoch}", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,  # unused: batch_sampler drives batching
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=4,
        save_only_model=True,
        gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        accelerator_config={"dispatch_batches": False},
    )

    trainer = BatchSamplerTrainer(
        model=model,
        args=targs,
        train_dataset=examples,
        batch_sampler=sampler,
        pad_id=tok.pad_token_id or 0,
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the processor so vLLM can load the multimodal config unchanged
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # pragma: no cover
        print("processor save skipped:", e)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
