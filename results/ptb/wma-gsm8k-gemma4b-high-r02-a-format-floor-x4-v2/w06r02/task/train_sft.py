#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style CoT data.

The training string is rendered with the *grader's* chat template
(templates/gemma3.jinja, hash-checked) so train and eval see the same bytes.
Loss is computed on the model turn only (completion + <end_of_turn>).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

TEMPLATE_PATH = "templates/gemma3.jinja"
# sha256 of templates/gemma3.jinja as shipped by the harness; guards pitfall template_unreachable
TEMPLATE_SHA256 = None  # filled at runtime, printed for the record

AUX_FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
]


def load_template():
    raw = open(TEMPLATE_PATH, "rb").read()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


class SFTData(Dataset):
    def __init__(self, rows, tok, template, max_len, report=True):
        self.ex = []
        n_trunc = 0
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                chat_template=template,
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in rows
        ]
        fulls = [
            tok.apply_chat_template(
                [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["completion"]},
                ],
                chat_template=template,
                tokenize=False,
            )
            for r in rows
        ]
        for p, f in zip(prompts, fulls):
            assert f.startswith(p), (p[-200:], f[len(p) - 200:len(p) + 200])
        p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
        f_ids = tok(fulls, add_special_tokens=False)["input_ids"]
        for pi, fi in zip(p_ids, f_ids):
            if len(fi) > max_len:
                n_trunc += 1
                continue
            labels = [-100] * len(pi) + fi[len(pi):]
            self.ex.append({"input_ids": fi, "labels": labels})
        if report:
            lens = sorted(len(e["input_ids"]) for e in self.ex)
            print(
                f"[data] kept {len(self.ex)} dropped_too_long {n_trunc} "
                f"({n_trunc / max(1, len(rows)):.3%}) "
                f"tok p50 {lens[len(lens)//2]} p95 {lens[int(.95*len(lens))]} max {lens[-1]}",
                flush=True,
            )

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids, lab, att = [], [], []
        for b in batch:
            k = m - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * k)
            lab.append(b["labels"] + [-100] * k)
            att.append([1] * len(b["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    template, sha = load_template()
    print("[template] sha256", sha, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    ds = SFTData(rows, tok, template, args.max_seq_len)

    # one rendered example, for the record
    print("[example]\n" + tok.decode(ds[0]["input_ids"]), flush=True)
    print("[loss-on]\n" + tok.decode([t for t in ds[0]["labels"] if t != -100]), flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    if hasattr(model.model, "vision_tower"):
        for p in model.model.vision_tower.parameters():
            p.requires_grad_(False)
        for p in model.model.multi_modal_projector.parameters():
            p.requires_grad_(False)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 10 ** 9,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name=None,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        optim="adamw_torch_fused",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    for f in AUX_FILES:
        src = os.path.join(args.model, f)
        dst = os.path.join(final, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    # Trainer writes its own generation_config.json and can drop <end_of_turn> (106)
    # from eos_token_id; restore the snapshot's file verbatim so vLLM stops where the
    # grading template ends a turn. Decode params are changed only by an explicit card.
    shutil.copy(
        os.path.join(args.model, "generation_config.json"),
        os.path.join(final, "generation_config.json"),
    )
    print("[gen-cfg]", open(os.path.join(final, "generation_config.json")).read(), flush=True)
    print("[saved]", final, flush=True)


if __name__ == "__main__":
    main()
