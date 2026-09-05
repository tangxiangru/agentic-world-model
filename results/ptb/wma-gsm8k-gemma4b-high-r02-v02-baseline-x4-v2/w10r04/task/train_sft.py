#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on prompt/completion jsonl.

The jsonl rows carry `prompt` (already rendered with templates/gemma3.jinja,
WITHOUT the leading <bos>, which the tokenizer adds) and `completion` (chain of
thought + "ANSWER: n" + <end_of_turn>).  Loss is on completion tokens only.

Batching is by token budget over length-sorted rows, so a micro-batch is
~constant in tokens regardless of sequence length; the vision tower is frozen.
"""
import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

BOS = "<bos>"


def tokenize_rows(rows, tok, max_len, verbose=True):
    out = []
    n_trunc = 0
    for r in rows:
        prompt = r["prompt"]
        if prompt.startswith(BOS):
            prompt = prompt[len(BOS):]
        p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
        c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        ids = p_ids + c_ids
        if len(ids) > max_len:
            n_trunc += 1
            continue
        out.append({"input_ids": ids, "n_prompt": len(p_ids)})
    if verbose:
        print(f"tokenized {len(out)} rows, dropped {n_trunc} over max_len={max_len}")
    return out


class TokenBudgetBatches(Dataset):
    """Pre-collated micro-batches: rows sorted by length, packed to a token budget."""

    def __init__(self, examples, budget, pad_id, max_seqs=64, seed=0):
        self.pad_id = pad_id
        order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
        batches, cur, cur_max = [], [], 0
        for i in order:
            L = len(examples[i]["input_ids"])
            m = max(cur_max, L)
            if cur and (m * (len(cur) + 1) > budget or len(cur) + 1 > max_seqs):
                batches.append(cur)
                cur, cur_max = [i], L
            else:
                cur.append(i)
                cur_max = m
        if cur:
            batches.append(cur)
        random.Random(seed).shuffle(batches)
        self.batches = batches
        self.examples = examples
        tot = sum(len(examples[i]["input_ids"]) for i in range(len(examples)))
        padded = sum(max(len(examples[i]["input_ids"]) for i in b) * len(b) for b in batches)
        print(f"{len(batches)} micro-batches, {tot/1e6:.1f}M real tokens, "
              f"{padded/1e6:.1f}M padded ({100*(padded-tot)/padded:.1f}% pad)")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = self.batches[i]
        rows = [self.examples[j] for j in idx]
        L = max(len(r["input_ids"]) for r in rows)
        input_ids, labels, attn = [], [], []
        for r in rows:
            ids = r["input_ids"]
            pad = L - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            lab = [-100] * r["n_prompt"] + ids[r["n_prompt"]:] + [-100] * pad
            labels.append(lab)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--no-liger", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data)]
    if a.limit:
        rows = rows[: a.limit]
    ex = tokenize_rows(rows, tok, a.max_seq_len)
    ds = TokenBudgetBatches(ex, a.token_budget, tok.pad_token_id, seed=a.seed)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    if not a.no_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(model=model)
        print("liger applied")
    # text-only corpus: the vision stack never sees a gradient
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e9:.2f}B")

    args = TrainingArguments(
        output_dir=a.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=a.warmup,
        weight_decay=a.weight_decay,
        bf16=True,
        logging_steps=20,
        save_strategy=a.save_strategy,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=a.optim,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to=[],
        seed=a.seed,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=lambda feats: feats[0],
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(os.path.join(a.out, "final"))
    tok.save_pretrained(os.path.join(a.out, "final"))
    print("saved", os.path.join(a.out, "final"))


if __name__ == "__main__":
    main()
