#!/usr/bin/env python3
"""Completion-only SFT for google/gemma-3-4b-pt on GSM8K-style data.

Rendering is done by hand rather than by the tokenizer's own chat template,
because the grader passes templates/gemma3.jinja to vLLM and the tokenizer that
ships with the checkpoint is not guaranteed to match it. `render()` is checked
byte-for-byte against that jinja file by verify_render.py before every launch.

Loss is masked over the prompt; the trailing <end_of_turn> is kept in the loss
so the model learns to stop where vLLM stops (eos_token_id = [1, 106]).
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

BOS = "<bos>"
SOT = "<start_of_turn>"
EOT = "<end_of_turn>"


def render(prompt: str, completion: str | None) -> str:
    """Exactly what templates/gemma3.jinja produces for
    [user: prompt] (+ [assistant: completion]), including add_generation_prompt.

    `completion` is expected to already carry the terminator, so that the
    stop_token preflight check reads the same bytes the model is trained on
    rather than a field the trainer silently decorates later."""
    s = f"{BOS}{SOT}user\n{prompt.strip()}{EOT}\n{SOT}model\n"
    if completion is not None:
        assert completion.endswith(EOT), "training targets must carry the stop token"
        s += completion
    return s


class SFTData(Dataset):
    def __init__(self, path: str, tok, max_seq_len: int, limit: int | None = None):
        self.rows = []
        n_dropped = 0
        lens = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                prefix = render(r["prompt"], None)
                full = render(r["prompt"], r["completion"])
                p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
                f_ids = tok(full, add_special_tokens=False)["input_ids"]
                if len(f_ids) > max_seq_len:
                    n_dropped += 1
                    continue
                labels = list(f_ids)
                labels[: len(p_ids)] = [-100] * len(p_ids)
                self.rows.append({"input_ids": f_ids, "labels": labels})
                lens.append(len(f_ids))
        lens.sort()
        self.stats = {
            "n": len(self.rows),
            "dropped_too_long": n_dropped,
            "p50_tokens": lens[len(lens) // 2] if lens else 0,
            "p99_tokens": lens[int(len(lens) * 0.99)] if lens else 0,
            "max_tokens": lens[-1] if lens else 0,
            "total_tokens": sum(lens),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class TokenBudgetBatches:
    """Length-grouped batches capped by *padded token count*, not row count.

    gemma-3's vocabulary is 262k, so the lm_head logits are ~1 MB per token in
    fp32 and they, not the weights, decide what fits: a fixed 8-row batch is
    3.2k tokens on a zero-shot row and 19k on a 10-shot row, and the second one
    OOMs an 80 GB card. Capping padded tokens per micro-batch makes memory flat
    across the length distribution.
    """

    def __init__(self, lengths, max_tokens: int, max_rows: int, seed: int, chunk: int = 4096):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.max_rows = max_rows
        self.seed = seed
        self.chunk = chunk
        self.epoch = 0
        self._batches = self._build(seed)

    def _build(self, seed):
        import random as _r

        rng = _r.Random(seed)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for start in range(0, len(idx), self.chunk):
            block = sorted(idx[start : start + self.chunk], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in block:
                new_max = max(cur_max, self.lengths[i])
                if cur and ((len(cur) + 1) * new_max > self.max_tokens or len(cur) >= self.max_rows):
                    batches.append(cur)
                    cur, cur_max = [i], self.lengths[i]
                else:
                    cur, cur_max = cur + [i], new_max
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._batches

    def __len__(self):
        return len(self._batches)


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, mask = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


class BudgetTrainer(Trainer):
    """Trainer whose train dataloader uses TokenBudgetBatches."""

    def set_batch_sampler(self, sampler):
        self._batch_sampler = sampler

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-tokens-per-batch", type=int, default=4096)
    ap.add_argument("--max-rows-per-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTData(args.data, tok, args.max_seq_len, args.limit)
    print("DATA STATS", json.dumps(ds.stats), flush=True)
    assert ds.stats["dropped_too_long"] <= 0.02 * (ds.stats["n"] + ds.stats["dropped_too_long"]), (
        "more than 2% of rows exceed max_seq_len"
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa"
    )
    # text-only task: the siglip tower and its projector never see a gradient,
    # and freezing them keeps them out of the optimizer state
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    model.config.use_cache = False

    batch_sampler = TokenBudgetBatches(
        [len(r["input_ids"]) for r in ds.rows],
        max_tokens=args.max_tokens_per_batch,
        max_rows=args.max_rows_per_batch,
        seed=args.seed,
    )
    print(
        f"BATCHES {len(batch_sampler)} micro-batches, "
        f"~{len(batch_sampler)//args.grad_accum} optimizer steps/epoch",
        flush=True,
    )

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=args.save_steps if args.save_steps else 1_000_000,
        save_strategy="steps" if args.save_steps else "no",
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        accelerator_config={"split_batches": False, "dispatch_batches": False},
        dataloader_num_workers=2,
        max_grad_norm=1.0,
        optim=args.optim,
    )

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        processing_class=tok,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.set_batch_sampler(batch_sampler)
    result = trainer.train()
    print("TRAIN RESULT", result.metrics, flush=True)

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    # train in fp32 (bf16 autocast) but ship bf16: the grader loads final_model/
    # with vLLM and the base checkpoint's own config.json says bfloat16
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.torch_dtype = "bfloat16"
    trainer.save_model(final)
    tok.save_pretrained(final)
    # vLLM loads Gemma3ForConditionalGeneration and wants the processor files too
    for extra in ("preprocessor_config.json", "processor_config.json", "added_tokens.json"):
        src = os.path.join(args.model, extra)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, extra)):
            with open(src, "rb") as a, open(os.path.join(final, extra), "wb") as b:
                b.write(a.read())
    with open(os.path.join(final, "train_stats.json"), "w") as f:
        json.dump({"data": ds.stats, "metrics": result.metrics, "args": vars(args)}, f, indent=2)
    print("SAVED", final, flush=True)


if __name__ == "__main__":
    main()
