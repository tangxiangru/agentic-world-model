#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered prompt/completion pairs.

The rows in --data are already rendered with the grader's chat template, so this
script tokenizes with add_special_tokens=False: the <bos> the template emitted is
the only <bos>. Loss is on completion tokens only.
"""
from __future__ import annotations
import argparse, json, math, os, random, shutil, sys, time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, set_seed)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class PC(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class CompletionOnlyTrainer(Trainer):
    """Gemma3ForConditionalGeneration.forward upcasts the FULL [B,T,262208] logit tensor
    to fp32 and then makes a second masked copy of it -- ~20 GB for a batch of 8 long rows,
    which OOMs an 80 GB H100. Prompt tokens carry no loss here (labels are -100 over the
    whole prompt, which is ~65% of tokens overall and ~90% of a 10-shot row), so we run the
    decoder ourselves and put only the supervised positions through the LM head."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        m = model.module if hasattr(model, "module") else model
        labels = inputs["labels"]
        out = m.model(input_ids=inputs["input_ids"],
                      attention_mask=inputs["attention_mask"])
        h = out[0][:, :-1, :]                       # [B, T-1, H]
        lab = labels[:, 1:]                         # next-token targets
        sel = lab.reshape(-1) != -100
        h_sel = h.reshape(-1, h.size(-1))[sel]
        lab_sel = lab.reshape(-1)[sel]
        logits = m.lm_head(h_sel)
        cap = getattr(m.config.text_config, "final_logit_softcapping", None)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        loss = F.cross_entropy(logits.float(), lab_sel, reduction="sum")
        # grad-accum-correct averaging: Trainer passes the token count across the whole
        # accumulated batch when it can, and multiplies our return value back by accum.
        loss = loss / (num_items_in_batch if num_items_in_batch else lab_sel.numel())
        return (loss, out) if return_outputs else loss


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ii, ll, am = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ii.append(b["input_ids"] + [pad_id] * k)
        ll.append(b["labels"] + [-100] * k)
        am.append([1] * len(b["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(am)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=3328)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(SNAP)
    pad_id = tok.pad_token_id

    raw = [json.loads(l) for l in open(args.data)]
    if args.limit:
        raw = raw[: args.limit]
    rows, dropped = [], 0
    t0 = time.time()
    P = tok([r["prompt"] for r in raw], add_special_tokens=False)["input_ids"]
    C = tok([r["completion"] for r in raw], add_special_tokens=False)["input_ids"]
    for p, c in zip(P, C):
        if len(p) + len(c) > args.max_seq_len:
            dropped += 1
            continue
        rows.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    print(f"tokenized {len(raw)} rows in {time.time()-t0:.0f}s; kept {len(rows)}, "
          f"dropped {dropped} ({dropped/max(1,len(raw)):.3%}) over max_seq_len={args.max_seq_len}",
          flush=True)
    assert dropped / max(1, len(raw)) < 0.02, "more than 2% of rows truncate"
    ntok = sum(len(r["input_ids"]) for r in rows)
    print(f"total tokens/epoch: {ntok/1e6:.1f}M  mean len {ntok/len(rows):.0f}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    print("model class:", type(model).__name__, flush=True)
    # text-only task: the SigLIP tower and projector never see a gradient
    frozen = 0
    for n_, p_ in model.named_parameters():
        if "vision_tower" in n_ or "multi_modal_projector" in n_:
            p_.requires_grad_(False)
            frozen += p_.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {frozen/1e6:.0f}M, trainable {trainable/1e9:.2f}B", flush=True)
    model.config.use_cache = False

    ta = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps if args.save_steps else 10**9,
        save_strategy="steps" if args.save_steps else "no",
        save_total_limit=2,
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        group_by_length=True,
        length_column_name=None,
        dataloader_num_workers=4,
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = CompletionOnlyTrainer(model=model, args=ta, train_dataset=PC(rows),
                                    data_collator=lambda b: collate(b, pad_id))
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # vLLM loads gemma3 through its processor; carry the vision-side configs across
    for f in ("preprocessor_config.json", "processor_config.json"):
        s = os.path.join(SNAP, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(final, f))
    # the grader stops on <end_of_turn> (106); keep it in the eos set
    gc = os.path.join(final, "generation_config.json")
    cfg = json.load(open(gc)) if os.path.exists(gc) else {}
    cfg.update({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                "cache_implementation": "hybrid"})
    json.dump(cfg, open(gc, "w"), indent=2)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
