#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style CoT.

Rows are rendered with templates/gemma3.jinja -- the *same* file evaluate.py
hands to vLLM -- so training and grading see byte-identical strings. Loss is
computed on the completion only; every target ends with <end_of_turn>.

    python train_sft.py --dry-run          # CPU: render + length stats, no GPU
    python train_sft.py --config ...       # the real run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

TASK_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = TASK_DIR / "templates" / "gemma3.jinja"
SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
STOP_TOKEN = "<end_of_turn>"


def load_template() -> tuple[str, str]:
    raw = TEMPLATE_PATH.read_bytes()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()[:12]


def build_messages(row: dict) -> list[dict]:
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    msgs.append({"role": "user", "content": row["prompt"]})
    return msgs


class SFTData(Dataset):
    def __init__(self, path, tok, template, max_len, limit=None, verbose=True):
        self.rows = []
        n_trunc = 0
        n_realign = 0
        lengths = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                msgs = build_messages(r)
                prompt_text = tok.apply_chat_template(
                    msgs, chat_template=template, tokenize=False,
                    add_generation_prompt=True,
                )
                comp = r["completion"].strip()
                assert comp.endswith(STOP_TOKEN), "completion must already carry the stop token"
                full_text = prompt_text + comp + "\n"
                p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
                f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
                if f_ids[: len(p_ids)] != p_ids:
                    # tokenizer merged across the boundary: find the real split
                    n_realign += 1
                    k = 0
                    while k < len(p_ids) and f_ids[k] == p_ids[k]:
                        k += 1
                    p_len = k
                else:
                    p_len = len(p_ids)
                lengths.append(len(f_ids))
                if len(f_ids) > max_len:
                    n_trunc += 1
                    continue
                self.rows.append((f_ids, p_len))
        if verbose:
            a = np.array(lengths)
            print(
                f"[data] {path}: kept {len(self.rows)}/{len(lengths)} rows; "
                f"len p50={np.percentile(a,50):.0f} p95={np.percentile(a,95):.0f} "
                f"p99={np.percentile(a,99):.0f} max={a.max()}; "
                f"dropped_over_{max_len}={n_trunc} ({100*n_trunc/len(a):.2f}%); "
                f"boundary_realign={n_realign}",
                flush=True,
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, p_len = self.rows[i]
        labels = list(ids)
        for j in range(min(p_len, len(labels))):
            labels[j] = -100
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", default="ckpts/exp-02")
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template, thash = load_template()
    print(f"[template] {TEMPLATE_PATH} sha256[:12]={thash}", flush=True)

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = SFTData(args.data, tok, template, args.max_seq_len, limit=args.limit)

    # show one rendered row exactly as the model will see it
    ids, p_len = ds.rows[0]
    print("=== PROMPT ===")
    print(tok.decode(ids[:p_len]))
    print("=== TARGET ===")
    print(tok.decode(ids[p_len:]))
    assert tok.decode(ids[p_len:]).rstrip().endswith(STOP_TOKEN), "target must end with the stop token"
    fs = next((i for i, (_, pl) in enumerate(ds.rows) if pl > 600), None)
    if fs is not None:
        print(f"=== FEW-SHOT ROW {fs} PROMPT HEAD ===")
        print(tok.decode(ds.rows[fs][0][: ds.rows[fs][1]])[:600])

    if args.dry_run:
        print("[dry-run] no GPU work done")
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_train/1e9:.2f}B, frozen {n_frozen/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        group_by_length=True,
        length_column_name=None,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    final = Path(args.out) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    trainer.model.save_pretrained(final, safe_serialization=True)
    AutoProcessor.from_pretrained(SNAPSHOT).save_pretrained(final)
    tok.save_pretrained(final)
    # the grader needs a generation_config that stops on <end_of_turn>
    gen = json.loads((Path(SNAPSHOT) / "generation_config.json").read_text())
    (final / "generation_config.json").write_text(json.dumps(gen, indent=2))
    print(f"[save] {final}", flush=True)


if __name__ == "__main__":
    main()
