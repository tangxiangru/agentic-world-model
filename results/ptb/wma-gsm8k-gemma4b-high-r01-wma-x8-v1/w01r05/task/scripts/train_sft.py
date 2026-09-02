#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style data.

Rows are rendered by scripts/render_sft.render, i.e. through the byte-identical
copy of the template the grader uses, and terminated with the token vLLM stops
on. Loss is taken on the completion only. The vision tower and multimodal
projector are frozen (no images anywhere in this task), which keeps the
architecture and therefore the saved config exactly what the grader can load.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_sft import STOP_TOKEN, render  # noqa: E402

SNAP = os.environ["PTB_BASE_MODEL_SNAPSHOT"]


class PackedRows(Dataset):
    def __init__(self, ids, plens):
        self.ids, self.plens = ids, plens

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "prompt_len": self.plens[i]}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids = torch.full((len(feats), n), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(feats), n), dtype=torch.long)
        labels = torch.full((len(feats), n), -100, dtype=torch.long)
        for i, f in enumerate(feats):
            x = f["input_ids"]
            L = len(x)
            input_ids[i, :L] = torch.tensor(x, dtype=torch.long)
            attn[i, :L] = 1
            p = f["prompt_len"]
            labels[i, p:L] = torch.tensor(x[p:], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def build(path: str, max_seq_len: int, limit: int | None):
    tok = AutoTokenizer.from_pretrained(SNAP)
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    prefixes, fulls = [], []
    for r in rows:
        p, fu = render(r["prompt"], r["completion"])
        prefixes.append(p)
        fulls.append(fu)
    enc_p = tok(prefixes, add_special_tokens=False)["input_ids"]
    enc_f = tok(fulls, add_special_tokens=False)["input_ids"]
    ids, plens, dropped = [], [], 0
    eot = tok.convert_tokens_to_ids(STOP_TOKEN)
    for p, fu in zip(enc_p, enc_f):
        if len(fu) > max_seq_len or len(fu) <= len(p):
            dropped += 1
            continue
        assert fu[: len(p)] == p, "prefix is not a token-prefix of the full sequence"
        assert fu[-1] == eot, "target does not end with the stop token"
        ids.append(fu)
        plens.append(len(p))
    lens = np.array([len(x) for x in ids])
    print(
        f"rows {len(ids)} (dropped {dropped}) tokens {lens.sum()/1e6:.1f}M "
        f"p50={np.percentile(lens,50):.0f} max={lens.max()}",
        flush=True,
    )
    return PackedRows(ids, plens), lens.tolist(), tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=1792)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    ds, lens, tok = build(args.data, args.max_seq_len, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.float32, attn_implementation=args.attn
    )
    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {frozen/1e9:.2f}B, trainable {trainable/1e9:.2f}B", flush=True)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.micro_bs,
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
        save_total_limit=3,
        report_to=[],
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim=args.optim,
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        save_safetensors=True,
    )
    # LengthGroupedSampler reads this off the dataset when the column is absent
    ds.length = lens  # type: ignore[attr-defined]

    class T(Trainer):
        def _get_train_sampler(self, *a, **k):
            from transformers.trainer_pt_utils import LengthGroupedSampler

            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lens,
            )

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            """Completion-only CE without ever materialising [B, T, 262144] logits.

            Gemma-3's vocabulary is 262144, so the default path allocates ~17 GB of
            fp32 logits for a batch of 16 and OOMs. Instead: run the backbone, gather
            only the positions that carry a label (~47% of tokens), and push those
            through lm_head in checkpointed chunks so each chunk's logits are freed
            after the forward and recomputed in backward.
            """
            labels = inputs["labels"]
            base = model.module if hasattr(model, "module") else model
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = base.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ).last_hidden_state
            h = hidden[:, :-1, :]
            lab = labels[:, 1:]
            keep = lab != -100
            hs = h[keep]
            ls = lab[keep]
            n_tok = hs.shape[0]
            head = base.lm_head

            def chunk_loss(x, y):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = head(x)
                return torch.nn.functional.cross_entropy(
                    logits.float(), y, reduction="sum"
                )

            total = hs.new_zeros((), dtype=torch.float32)
            CH = 2048
            for i in range(0, n_tok, CH):
                total = total + torch.utils.checkpoint.checkpoint(
                    chunk_loss, hs[i : i + CH], ls[i : i + CH], use_reentrant=False
                )
            denom = num_items_in_batch if num_items_in_batch is not None else n_tok
            return total / denom

    trainer = T(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    t0 = time.time()
    out = trainer.train()
    wall = time.time() - t0
    print(f"train wall {wall/3600:.2f} h; metrics {out.metrics}", flush=True)

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    # Save bf16: the grader loads this with vLLM, and the base checkpoint's own
    # config declares bfloat16. Training kept fp32 master weights; that is a
    # training-time choice and must not leak into the artifact.
    model.to(torch.bfloat16)
    trainer.save_model(final)
    tok.save_pretrained(final)
    with open(os.path.join(final, "train_summary.json"), "w") as f:
        json.dump(
            {
                "wall_h": wall / 3600,
                "metrics": out.metrics,
                "args": vars(args),
                "n_rows": len(ds),
            },
            f,
            indent=2,
        )
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
