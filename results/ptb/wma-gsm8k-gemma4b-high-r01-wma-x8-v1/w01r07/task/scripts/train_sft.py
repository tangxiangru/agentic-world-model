#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on pre-rendered prompt/completion text.

The jsonl rows already contain the exact strings the grader's chat template
produces (verified byte-for-byte against templates/gemma3.jinja), so nothing
here re-templates anything: we tokenize with add_special_tokens=False and mask
the prompt out of the loss.
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

SNAP = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)


class PackedRows(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_drop = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                self.rows.append(json.loads(line))
        prompts = [r["prompt"] for r in self.rows]
        comps = [r["completion"] for r in self.rows]
        p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
        c_ids = tok(comps, add_special_tokens=False)["input_ids"]
        self.examples = []
        for p, c in zip(p_ids, c_ids):
            if len(p) + len(c) > max_seq_len:
                n_drop += 1
                continue
            self.examples.append((p, c))
        self.lengths = [len(p) + len(c) for p, c in self.examples]
        print(f"[data] {len(self.examples)} rows kept, {n_drop} dropped over max_seq_len={max_seq_len}")
        print(f"[data] token total {sum(self.lengths)/1e6:.1f}M, max {max(self.lengths)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        p, c = self.examples[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


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
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--init-from", default=SNAP)
    ap.add_argument("--max-seq-len", type=int, default=3584)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    ds = PackedRows(args.data, tok, args.max_seq_len, args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init_from,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # text-only task: freeze the vision tower and the multimodal projector
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_train/1e9:.3f}B, frozen {n_frozen/1e9:.3f}B")
    model.config.use_cache = False
    # a greedy generation_config from a previous round would make every
    # checkpoint save raise "GenerationConfig is invalid"; neutralise it here
    # and rewrite the greedy file after the final save.
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_k = 64
    model.generation_config.top_p = 0.95

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=5,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim=args.optim,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # gemma3's vocab is 262k: the stock loss path materialises fp32 logits
        # (and a masked copy of them) and OOMs at bs=8. Liger's fused
        # linear+cross-entropy never materialises the full logit tensor.
        use_liger_kernel=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    final = os.path.join(args.output_dir, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the grader's decode deterministic
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(
            {
                "bos_token_id": 2,
                "cache_implementation": "hybrid",
                # do_sample must stay True or GenerationConfig.save_pretrained
                # rejects the temperature/top_k fields on the next round trip;
                # vLLM ignores do_sample and reads temperature==0 as greedy.
                "do_sample": True,
                "eos_token_id": [1, 106],
                "pad_token_id": 0,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0,
                "transformers_version": "4.50.0.dev0",
            },
            f,
            indent=2,
        )
    # the processor files vLLM wants for a multimodal gemma3 config
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            with open(src) as a, open(os.path.join(final, fn), "w") as b:
                b.write(a.read())
    print(f"[done] saved {final}")


if __name__ == "__main__":
    main()
