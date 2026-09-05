#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on GSM8K-style CoT.

Rows are rendered with the *grading* chat template (templates/gemma3.jinja,
hash-checked in scripts/evalfmt.py) so training and grading see the same string.
Every completion ends with the grading stop token <end_of_turn>, and every
target's last number is the answer -- that is what the grader reads.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalfmt as E  # noqa: E402

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, fewshot_frac, seed, limit=None):
        self.tok = tok
        self.max_seq_len = max_seq_len
        rng = random.Random(seed)
        rows = [json.loads(l) for l in open(path)]
        if limit:
            rows = rows[:limit]

        full_sys = E.fewshot_system_message()

        self.examples = []
        self.lengths = []
        n_trunc = 0
        eot_id = tok.convert_tokens_to_ids(E.STOP_TOKEN)
        assert eot_id is not None and eot_id > 0
        for r in rows:
            use_shots = rng.random() < fewshot_frac
            sys_msg = full_sys if use_shots else None
            prompt = tok.apply_chat_template(
                E.messages(r["question"], sys_msg),
                tokenize=False,
                add_generation_prompt=True,
            )
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(r["target"], add_special_tokens=False)["input_ids"]
            assert c_ids[-1] == eot_id, "target must end with the grading stop token"
            ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + list(c_ids)
            if len(ids) > max_seq_len:
                n_trunc += 1
                continue
            self.examples.append((ids, labels))
            self.lengths.append(len(ids))
        self.n_trunc = n_trunc
        print(f"[data] kept {len(self.examples)} rows, dropped {n_trunc} over max_seq_len "
              f"({n_trunc / max(1, len(rows)):.3%})", flush=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labels = self.examples[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class MaskedCETrainer(Trainer):
    """Cross-entropy over the labelled tokens only, without materialising logits.

    Gemma-3's vocabulary is 262144, so the default path builds a
    (batch, seq, 262144) fp32 logit tensor -- 19 GB for one micro-batch of long
    10-shot rows, which OOMs an 80 GB H100. Under completion-only loss only ~12%
    of positions carry a label, so gather those hidden states first and run
    lm_head on them alone. Same loss, ~1/8 the logit memory and compute.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        core = self.accelerator.unwrap_model(model)
        hidden = core.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state
        sel = labels[:, 1:] != -100
        h = hidden[:, :-1, :][sel]
        t = labels[:, 1:][sel]
        logits = core.lm_head(h).float()
        if num_items_in_batch is not None:
            loss = torch.nn.functional.cross_entropy(logits, t, reduction="sum")
            loss = loss / num_items_in_batch
        else:
            loss = torch.nn.functional.cross_entropy(logits, t)
        return (loss, None) if return_outputs else loss


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--greedy", action="store_true",
                    help="write a greedy generation_config.json instead of copying the base one")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.chat_template = E.chat_template()
    tok.padding_side = "right"

    ds = SFTRows(args.data, tok, args.max_seq_len, args.fewshot_frac, args.seed,
                 limit=args.limit)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    # text-only task: the SigLIP tower and its projector never see a gradient,
    # so freeze them and keep their optimizer state off the GPU.
    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] frozen {frozen/1e6:.0f}M, trainable {trainable/1e6:.0f}M", flush=True)
    # A parent trained by this script carries the greedy generation_config written
    # below, and transformers' save_pretrained validates generation configs
    # strictly: do_sample=False together with temperature/top_k raises and kills
    # the run at the first checkpoint. Reset to the base model's valid config for
    # the duration of training; the greedy file is rewritten after save_model.
    model.generation_config = GenerationConfig.from_pretrained(SNAPSHOT)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
    )

    trainer = MaskedCETrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    # compute_loss normalises by num_items_in_batch itself, so Trainer must not
    # also divide by gradient_accumulation_steps.
    trainer.model_accepts_loss_kwargs = True
    out = trainer.train()
    print(out, flush=True)

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # vLLM loads Gemma3ForConditionalGeneration; carry the processor files over.
    for f in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(SNAPSHOT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, f))
    # vLLM reads generation_config.json for its request defaults and inspect does
    # not set temperature, so this file decides the decode policy at grading time.
    # Default: keep the base model's file byte-for-byte, so an SFT card measures
    # SFT alone. --greedy is a separate intervention, measured on its own card.
    if args.greedy:
        gc = {
            "bos_token_id": 2,
            "eos_token_id": [1, 106],
            "pad_token_id": 0,
            "cache_implementation": "hybrid",
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
        }
        with open(os.path.join(final, "generation_config.json"), "w") as f:
            json.dump(gc, f, indent=2)
    else:
        shutil.copy(os.path.join(SNAPSHOT, "generation_config.json"),
                    os.path.join(final, "generation_config.json"))
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
