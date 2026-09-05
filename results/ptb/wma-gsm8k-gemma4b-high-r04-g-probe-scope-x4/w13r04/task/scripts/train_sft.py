#!/usr/bin/env python3
"""Prompt-masked SFT for google/gemma-3-4b-pt on gsm8k-style data.

Rows are pre-rendered by build_sft_data.py with the grader's own gemma3
template, so this script only tokenizes, masks the prompt, and trains.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


class SFTRows(Dataset):
    def __init__(self, path: str, tok, max_seq_len: int, limit: int | None = None):
        self.ex = []
        self.n_truncated = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                ids = p + c
                if len(ids) > max_seq_len:
                    self.n_truncated += 1
                    continue  # drop rather than truncate: a truncated target
                    # loses the stop token and the answer line
                labels = [-100] * len(p) + c[:]
                self.ex.append((ids, labels))
        self.lengths = [len(e[0]) for e in self.ex]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, labels = self.ex[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = ((n + 7) // 8) * 8
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Micro-batches with a fixed token budget instead of a fixed row count.

    Rows are 340 tokens at the median but 2500 for the 8% that carry the
    grader's 10-shot prefix. A fixed row count therefore either wastes the GPU
    on the short batches or blows the memory budget on the long ones; a token
    budget keeps every micro-batch the same size in work and in memory.
    """

    def __init__(self, lengths, budget, max_rows, seed):
        self.batches = []
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        cur, cur_max = [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and ((len(cur) + 1) * m > budget or len(cur) >= max_rows):
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur.append(i)
                cur_max = m
        if cur:
            self.batches.append(cur)
        random.Random(seed).shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _chunk_ce(hidden, labels, lm_head, chunk):
    """Cross-entropy over selected positions, in checkpointed chunks.

    gemma-3's vocabulary is 262144 wide: the logits for 11k supervised
    positions are 12 GB in fp32 before autograd keeps a second and third copy.
    Chunking under torch.utils.checkpoint bounds that to one chunk at a time.
    """

    def step(h, l):
        return F.cross_entropy(lm_head(h).float(), l, reduction="sum")

    n = hidden.size(0)
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        total = total + torch.utils.checkpoint.checkpoint(
            step, hidden[i : i + chunk], labels[i : i + chunk], use_reentrant=False
        )
    return total / max(1, n)


class MaskedLossTrainer(Trainer):
    """Compute the LM loss only at supervised positions.

    Gemma-3's vocabulary is 262144, so the default path materialises a
    [B, T, 262144] logits tensor plus an fp32 copy and a log-softmax save -
    ~36 GB for one 8x2512 micro-batch, on top of ~43 GB of weights and grads.
    Under completion-only loss only ~15% of positions carry a label, so the
    head is applied to those positions alone. The loss value is identical
    (token-mean over supervised positions); only the memory changes.
    """

    ce_chunk = 4096
    batch_sampler = None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        core = model.module if hasattr(model, "module") else model
        # accelerate wraps model.forward in autocast, not the submodules; this
        # method calls core.model directly, so the autocast context has to be
        # re-entered by hand or every GEMM runs in true fp32 (measured: 1.5k
        # tok/s instead of 7.8k).
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self._loss(core, inputs, labels)

    def _loss(self, core, inputs, labels):
        hidden = core.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )[0]
        shift_h = hidden[:, :-1, :]
        shift_l = labels[:, 1:]
        sel = shift_l != -100
        return _chunk_ce(shift_h[sel], shift_l[sel], core.lm_head, self.ce_chunk)

    def get_train_dataloader(self):
        if self.batch_sampler is None:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader

        dl = DataLoader(
            self.train_dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )
        return self.accelerator.prepare(dl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=3328)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--token-budget", type=int, default=24000)
    ap.add_argument("--max-rows-per-batch", type=int, default=64)
    ap.add_argument("--ce-chunk", type=int, default=4096)
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.limit)
    print(
        f"rows={len(ds)} dropped_over_{args.max_seq_len}={ds.n_truncated} "
        f"tokens={sum(ds.lengths)} p50={int(np.percentile(ds.lengths,50))} "
        f"max={max(ds.lengths)}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,  # fp32 master weights; bf16 autocast via TrainingArguments
        attn_implementation=args.attn,
    )
    model.config.use_cache = False
    # A parent checkpoint from an earlier card ships the greedy generation config
    # (do_sample false with temperature/top_p/top_k set). GenerationConfig.validate()
    # turns that combination into a ValueError inside save_pretrained, so the first
    # checkpoint save would kill the run. Neutralise it in memory; the explicit
    # greedy JSON is written next to the final weights at the end of this script.
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0
    )
    frozen = 0
    for n, p in model.named_parameters():
        if n.startswith("model.vision_tower") or n.startswith(
            "model.multi_modal_projector"
        ):
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen={frozen/1e9:.2f}B trainable={trainable/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        report_to=[],
        group_by_length=False,
        remove_unused_columns=False,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )

    trainer = MaskedLossTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.ce_chunk = args.ce_chunk
    trainer.batch_sampler = TokenBudgetBatches(
        ds.lengths, args.token_budget, args.max_rows_per_batch, args.seed
    )
    print(
        f"micro_batches={len(trainer.batch_sampler)} "
        f"rows_per_batch p50={int(np.percentile([len(b) for b in trainer.batch_sampler.batches],50))} "
        f"max={max(len(b) for b in trainer.batch_sampler.batches)}",
        flush=True,
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.model.to(torch.bfloat16)
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    # the grader loads final_model/ with vLLM as Gemma3ForConditionalGeneration:
    # it needs the processor configs too (pitfalls.yaml final_model_not_loadable)
    import shutil
    for fn in ("preprocessor_config.json", "processor_config.json",
               "added_tokens.json", "special_tokens_map.json", "tokenizer.model"):
        src = os.path.join(args.model, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, fn)):
            shutil.copy(src, os.path.join(final, fn))
    # decode config the grader will pick up: greedy, stop on <end_of_turn>
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(
            {
                "bos_token_id": 2,
                "eos_token_id": [1, 106],
                "pad_token_id": 0,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "transformers_version": "4.57.3",
            },
            f,
            indent=2,
        )
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
