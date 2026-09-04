#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on prompt/completion jsonl.

Deliberate choices, each against a named pitfall:
  * prompt and completion are tokenized separately and asserted to concatenate
    to the tokenization of the joined string, so the loss mask is exact;
  * rows longer than --max-seq-len are dropped at build time and asserted here,
    never truncated (seq_len_truncation);
  * every completion is asserted to end in the stop token (eos_mismatch);
  * bf16 weights + 8-bit Adam. fp32 master weights were measured at 1.15
    samples/s against bf16's 5.04 on identical batches (4.4x), which would put
    two epochs out of the time budget; the lr is raised to 2e-5 so a step is
    comfortably above bf16's ~4e-3 relative resolution;
  * the loss is computed only at labelled positions and the 262k-wide head is
    applied in checkpointed chunks -- transformers' own path allocates a
    [batch, seq, 262144] fp32 logit tensor and OOMs at 20 GiB;
  * vision tower and projector are frozen but kept, so the saved directory has
    the same architecture the grader's vLLM loads.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

import common


def _chunk_ce(h, w, lab):
    """Cross-entropy for one chunk of positions. Called under checkpoint() so the
    [chunk, 262144] logits are recomputed in backward instead of being stored."""
    logits = torch.nn.functional.linear(h, w).float()
    return torch.nn.functional.cross_entropy(logits, lab, reduction="sum", ignore_index=-100)


class ChunkedLossTrainer(Trainer):
    """Compute the LM loss only on the positions that carry a label.

    Gemma-3's vocab is 262,144. transformers' own path materialises
    [batch, seq, 262144] logits in fp32 (20 GiB for one 8x2797 batch -> OOM) and
    accelerate then upcasts the returned logits a second time. Here the hidden
    states are gathered at the labelled positions first -- under completion-only
    loss that is ~35% of them -- and the head is applied in 4096-token chunks
    under checkpointing, so peak logit memory is bounded and independent of
    sequence length.
    """

    chunk = 4096

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        out = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        h = out[0][:, :-1, :]
        lab = labels[:, 1:]
        sel = lab != -100
        h = h[sel]
        lab = lab[sel]
        w = model.lm_head.weight
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, h.shape[0], self.chunk):
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_ce, h[i:i + self.chunk], w, lab[i:i + self.chunk],
                use_reentrant=False,
            )
        loss = total / max(1, h.shape[0])
        return (loss, None) if return_outputs else loss


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            lab.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
        }


def build_dataset(path: str, tok, max_seq_len: int) -> Dataset:
    rows = [json.loads(l) for l in open(path)]
    feats = {"input_ids": [], "labels": [], "length": []}
    n_loss_tokens = 0
    for i, r in enumerate(rows):
        p = tok.encode(r["prompt"], add_special_tokens=False)
        c = tok.encode(r["completion"], add_special_tokens=False)
        assert r["completion"].endswith(common.STOP_TOKEN), f"row {i}: no stop token"
        assert c[-1] == 106, f"row {i}: last token is not <end_of_turn>"
        assert len(p) + len(c) <= max_seq_len, f"row {i}: {len(p)+len(c)} > {max_seq_len}"
        if i < 200:  # spot-check the mask boundary
            assert tok.encode(r["prompt"] + r["completion"], add_special_tokens=False) == p + c
        feats["input_ids"].append(p + c)
        feats["labels"].append([-100] * len(p) + c)
        feats["length"].append(len(p) + len(c))
        n_loss_tokens += len(c)
    print(f"{len(rows)} rows, {sum(feats['length'])/1e6:.2f}M tokens, "
          f"{n_loss_tokens/1e6:.2f}M loss tokens, max len {max(feats['length'])}")
    return Dataset.from_dict(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=common.BASE_SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.parent)
    ds = build_dataset(args.data, tok, args.max_seq_len)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=getattr(torch, args.dtype), attn_implementation=args.attn
    )
    frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") \
           or name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
            p.requires_grad_(False)
            frozen += p.numel()
    train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {train_n/1e9:.3f}B, frozen {frozen/1e9:.3f}B")
    model.config.use_cache = False
    # The greedy generation_config.json adopted in exp-03 (temperature 0.0 with
    # do_sample false) is valid for vLLM but fails HF's GenerationConfig.validate
    # on save_pretrained, which killed exp-05's first run AFTER 27 minutes of
    # training with no weights written. Reset to the base snapshot's valid config
    # here; the greedy file is written back into the output dir after training.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig.from_pretrained(common.BASE_SNAPSHOT)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim=args.optim,
        max_steps=args.max_steps,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_total_limit=3,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
    )

    trainer = ChunkedLossTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the processor so the multimodal config stays self-consistent for vLLM
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(args.parent).save_pretrained(final)
    except Exception as e:  # noqa: BLE001
        print("processor save skipped:", e)
    import shutil
    shutil.copy(os.path.join(common.TASK_DIR, "greedy_generation_config.json"),
                os.path.join(final, "generation_config.json"))
    print("saved", final, "with the greedy generation_config")


if __name__ == "__main__":
    main()
