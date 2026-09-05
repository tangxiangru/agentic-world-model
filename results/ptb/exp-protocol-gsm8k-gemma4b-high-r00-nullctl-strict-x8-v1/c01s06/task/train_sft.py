#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on GSM8K-style math CoT data.

Trains in the exact chat format that templates/gemma3.jinja renders at eval time,
with the loss restricted to the assistant turn.
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

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

IGNORE = -100


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, max_prompt_len):
        self.ex = []
        n_skip = 0
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        for r in rows:
            p = f"<bos><start_of_turn>user\n{r['prompt'].strip()}<end_of_turn>\n<start_of_turn>model\n"
            c = f"{r['completion'].strip()}<end_of_turn>\n"
            pi = tok(p, add_special_tokens=False)["input_ids"]
            ci = tok(c, add_special_tokens=False)["input_ids"]
            if len(pi) > max_prompt_len or len(pi) + len(ci) > max_len:
                n_skip += 1
                continue
            self.ex.append((pi, ci))
        print(f"dataset: {len(self.ex)} examples ({n_skip} skipped for length)")
        self.lengths = [len(a) + len(b) for a, b in self.ex]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        pi, ci = self.ex[i]
        ids = pi + ci
        labels = [IGNORE] * len(pi) + ci[:]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = m - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [IGNORE] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
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
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-prompt-len", type=int, default=768)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTData(args.data, tok, args.max_len, args.max_prompt_len)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.float32, attn_implementation=args.attn
    )
    # fused linear cross-entropy: never materialises the [B, T, 262144] logits tensor
    from liger_kernel.transformers import apply_liger_kernel_to_gemma3
    apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True, model=model)
    # text-only task: freeze the vision stack
    for n, p in model.named_parameters():
        if n.startswith("model.vision_tower") or n.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    model.config.use_cache = False
    # a greedy generation_config.json (written at the end) fails HF validation,
    # so keep a plain one on the model while training/saving
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid",
    )

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 10**9,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        report_to=[],
        seed=17,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    model = model.to(torch.bfloat16)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    # copy processor files so vLLM can load the multimodal config
    import shutil
    for fn in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))
    # ship greedy decoding as the model's default sampling params
    with open(os.path.join(args.out, "generation_config.json"), "w") as fh:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
                   "cache_implementation": "hybrid"}, fh, indent=2)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
