#!/usr/bin/env python3
"""Full fine-tune of gemma-3-4b-pt on pre-rendered prompt/completion pairs."""
import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, limit=None, drop_exclude=None):
        self.ex = []
        n_long = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and len(self.ex) >= limit:
                    break
                r = json.loads(line)
                if drop_exclude and r["question"].strip() in drop_exclude:
                    continue
                p = tok(r["prompt_text"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion_text"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_len:
                    n_long += 1
                    continue
                ids = p + c
                labels = [-100] * len(p) + c[:]
                self.ex.append((ids, labels))
        print(f"loaded {len(self.ex)} examples ({n_long} dropped for length)")

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, labels = self.ex[i]
        return {"input_ids": ids, "labels": labels}


class FusedCELossTrainer(Trainer):
    """Compute the LM loss with liger's fused linear+CE to avoid materialising
    the [B, T, 262208] logit tensor (which dominates memory for Gemma-3)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )
        self._fused = LigerFusedLinearCrossEntropyFunction.apply

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.module if hasattr(model, "module") else model
        labels = inputs.pop("labels")
        out = base.model.language_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        h = out.last_hidden_state[:, :-1, :]
        tgt = labels[:, 1:]
        mask = tgt != -100
        h = h[mask]
        tgt = tgt[mask]
        loss = self._fused(h, base.lm_head.weight, tgt, None, None, -100, 0.0, 0.0, "mean")
        if isinstance(loss, tuple):
            loss = loss[0]
        return (loss, out) if return_outputs else loss


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * n)
            labels.append(f["labels"] + [-100] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--out", default="ckpt/sft1")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTData(args.data, tok, args.max_len, args.limit)

    model = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # our checkpoints ship a greedy generation_config (temperature=0) for vLLM,
    # which HF refuses to re-serialise; restore a valid one for save_pretrained
    # (the greedy file is rewritten by hand at the end).
    if model.generation_config is not None:
        model.generation_config.do_sample = True
        model.generation_config.temperature = 1.0
        model.generation_config.top_k = 64
        model.generation_config.top_p = 0.95
    # freeze the vision tower / multimodal projector: we only train text
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen params: {n_frozen/1e6:.1f}M")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        group_by_length=True,
        length_column_name=None,
        dataloader_num_workers=2,
        seed=args.seed,
    )

    trainer = FusedCELossTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    try:
        trainer.save_model(args.out)
    except Exception as e:  # never lose a finished run to a serialisation quirk
        print("save_model failed, falling back to save_pretrained:", e)
        model.generation_config = None
        model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    # copy remaining assets so vLLM can load it as a full gemma-3 multimodal model
    import shutil
    for fn in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))
    # greedy decoding at eval time
    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    json.dump(gc, open(os.path.join(args.out, "generation_config.json"), "w"), indent=2)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
