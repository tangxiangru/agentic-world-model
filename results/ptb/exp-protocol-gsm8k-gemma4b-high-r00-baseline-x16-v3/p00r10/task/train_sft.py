#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on jsonl rows of {prompt, completion}.

The rows are already rendered by fmt.py through the grader's own chat template, so
this script does no templating: it tokenizes prompt and completion separately,
masks the prompt out of the loss, and never adds special tokens (the <bos> is
already in the prompt string).
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
END_OF_TURN_ID = 106


class JsonlSFT(Dataset):
    def __init__(self, path: str, tok, max_seq_len: int):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                assert END_OF_TURN_ID in c[-3:], "target does not end with <end_of_turn>"
                if len(p) + len(c) > max_seq_len:
                    n_trunc += 1
                    continue
                self.rows.append((p, c))
        self.lengths = [len(p) + len(c) for p, c in self.rows]
        print(f"{path}: {len(self.rows)} rows kept, {n_trunc} dropped for exceeding max_seq_len")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        return {
            "input_ids": p + c,
            "labels": [-100] * len(p) + c,
            "length": len(p) + len(c),
        }


class TokenBudgetBatches:
    """Batches sized so that padded_tokens = max_len * n_rows stays under a budget.

    Fixed row counts OOM here: the vocab is 262k, so cross-entropy upcasts
    (rows x seq_len x 262144) floats. A 2.5k-token row at batch 8 alone wants 21 GiB.
    Bounding padded tokens instead bounds that allocation directly, and sorting by
    length first means short rows still travel in big batches.
    """

    def __init__(self, lengths, budget: int, seed: int):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches, cur, cur_max = [], [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and m * (len(cur) + 1) > budget:
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur, cur_max = cur + [i], m
        if cur:
            self.batches.append(cur)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        import random

        order = list(range(len(self.batches)))
        random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        for b in order:
            yield self.batches[b]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class BudgetTrainer(Trainer):
    def __init__(self, *a, batch_sampler=None, **kw):
        super().__init__(*a, **kw)
        self._batch_sampler = batch_sampler

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader

        return self.accelerator.prepare(
            DataLoader(
                self.train_dataset,
                batch_sampler=self._batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = JsonlSFT(args.data, tok, args.max_seq_len)
    print(f"total training tokens: {sum(ds.lengths) / 1e6:.1f}M")

    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.parent, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    except Exception as e:  # pragma: no cover
        print("flash_attention_2 unavailable, falling back to eager:", e)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.parent, dtype=torch.bfloat16, attn_implementation="eager"
        )
    model.config.use_cache = False
    # no images in this data: the vision stack gets no gradient, so keep it out of AdamW
    if hasattr(model.model, "vision_tower"):
        model.model.vision_tower.requires_grad_(False)
        model.model.multi_modal_projector.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable / 1e9:.2f}B")

    sampler = TokenBudgetBatches(ds.lengths, args.token_budget, args.seed)
    print(f"{len(sampler)} micro-batches/epoch, {len(sampler) // args.grad_accum} optimizer steps")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps if args.save_steps else 10**9,
        save_strategy="steps" if args.save_steps else "no",
        save_total_limit=None,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        accelerator_config={"split_batches": False, "dispatch_batches": False},
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        save_safetensors=True,
    )

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
        batch_sampler=sampler,
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # vLLM reads generation_config for the stop ids; keep the base one verbatim
    for name in ("generation_config.json", "preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAPSHOT, name)
        if os.path.exists(src):
            with open(src) as f, open(os.path.join(final, name), "w") as g:
                g.write(f.read())
    print("saved", final)


if __name__ == "__main__":
    main()
