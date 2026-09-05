"""Completion-only SFT for google/gemma-3-4b-pt on GSM8K-style data.

Renders every row with the *grader's* template (templates/gemma3.jinja) so the
training string is byte-identical to what vLLM will see, masks the prompt, and
ends every target with <end_of_turn> (eos id 106, the token vLLM stops on).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_utils as P  # noqa: E402

IGNORE = -100


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, fewshot_frac=0.0, seed=0, limit=None):
        self.rows = [json.loads(l) for l in open(path)]
        if limit:
            self.rows = self.rows[:limit]
        self.tok = tok
        self.max_seq_len = max_seq_len
        self.fewshot_frac = fewshot_frac
        self.rng = random.Random(seed)
        self.eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
        self.n_trunc = 0
        self.flags = [self.rng.random() < fewshot_frac for _ in self.rows]

    def __len__(self):
        return len(self.rows)

    def build(self, i):
        r = self.rows[i]
        prompt = P.eval_prompt(self.tok, r["question"], fewshot=self.flags[i])
        # prompt already ends with "<start_of_turn>model\n"; the target is the
        # solution followed by the stop token the grading template stops on.
        target = r["completion"]  # already ends with <end_of_turn> (see build_data.py)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        return p_ids, t_ids

    def __getitem__(self, i):
        p_ids, t_ids = self.build(i)
        ids = p_ids + t_ids
        labels = [IGNORE] * len(p_ids) + list(t_ids)
        if len(ids) > self.max_seq_len:  # keep the target, drop the prompt head
            over = len(ids) - self.max_seq_len
            ids = ids[over:]
            labels = labels[over:]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [IGNORE] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.fewshot_frac, args.seed, args.limit)

    # --- pre-flight on the actual rows -------------------------------------
    lens = []
    probe = range(0, len(ds), max(1, len(ds) // 2000))
    for i in probe:
        p, t = ds.build(i)
        lens.append(len(p) + len(t))
        assert t[-1] == ds.eot_id, f"row {i} target does not end with <end_of_turn>"
    lens.sort()
    trunc = sum(1 for L in lens if L > args.max_seq_len) / len(lens)
    print(f"[preflight] sampled {len(lens)} rows: p50={lens[len(lens)//2]} "
          f"p99={lens[int(len(lens)*0.99)]} max={lens[-1]} "
          f"truncated={trunc:.3%} (max_seq_len={args.max_seq_len})")
    ex = ds[0]
    print("[preflight] example rendering (last 400 chars of the loss region):")
    print(repr(tok.decode([t for t in ex["labels"] if t != IGNORE])[-400:]))
    if args.dry_run:
        return
    assert trunc < 0.02, f"{trunc:.3%} of rows truncate; raise --max-seq-len"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model.config.use_cache = False
    # text-only training: freeze the vision tower so it is saved unchanged
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"[model] froze {n_frozen/1e6:.0f}M vision params; "
          f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        group_by_length=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id or 0))
    res = trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the processor so vLLM can load the multimodal config unchanged
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # noqa: BLE001
        print("processor save skipped:", e)
    with open(os.path.join(args.out, "train_summary.json"), "w") as f:
        json.dump({"final_loss": res.training_loss, "steps": res.global_step,
                   "args": vars(args)}, f, indent=1)
    print("saved", final, res.training_loss, res.global_step)


if __name__ == "__main__":
    main()
