#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on GSM8K-style chain-of-thought data.

Prompts are rendered with templates/gemma3.jinja -- the same file evaluate.py
hands to vLLM -- and loss is taken on the completion only.  The completion
always ends with <end_of_turn>, the token the grader stops on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import common_fmt

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def get_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    tok.chat_template = common_fmt.load_template()
    return tok


def encode_rows(rows, tok, max_seq_len: int, verbose: bool = True):
    feats, n_trunc, lens, plens = [], 0, [], []
    stop_id = tok.convert_tokens_to_ids(common_fmt.STOP_TOKEN)
    assert stop_id is not None and stop_id > 0, stop_id
    for r in rows:
        shots = [tuple(s) for s in r.get("shots") or []]
        msgs = common_fmt.build_messages(r["question"], shots)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        tgt = r["target"]
        if not tgt.endswith(common_fmt.STOP_TOKEN):
            tgt = tgt.strip() + common_fmt.STOP_TOKEN
        completion = tgt
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        assert c_ids[-1] == stop_id, "completion does not end with the stop token"
        ids = p_ids + c_ids
        lens.append(len(ids))
        plens.append(len(p_ids))
        if len(ids) > max_seq_len:
            n_trunc += 1
            continue
        labels = [-100] * len(p_ids) + c_ids
        feats.append({"input_ids": ids, "labels": labels})
    if verbose:
        a = np.array(lens)
        print(f"rows={len(rows)} kept={len(feats)} dropped_over_len={n_trunc} "
              f"({n_trunc / max(1, len(rows)):.3%})")
        print(f"total len p50={np.percentile(a,50):.0f} p90={np.percentile(a,90):.0f} "
              f"p99={np.percentile(a,99):.0f} max={a.max()}")
        print(f"prompt len p50={np.percentile(plens,50):.0f} max={max(plens)}")
        print(f"total tokens kept={sum(len(f['input_ids']) for f in feats):,}")
    return feats


def token_budget_batches(lengths, budget: int, max_bs: int, seed: int):
    """Group examples of similar length into batches of <= `budget` padded tokens.

    gemma-3's 262k vocab makes the logits tensor the memory bottleneck
    (batch_tokens * 262144 * 2 bytes), so a fixed example count OOMs on the
    10-shot rows while wasting the GPU on the short ones.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and ((len(cur) + 1) * m > budget or len(cur) >= max_bs):
            batches.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur.append(i)
            cur_max = m
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * k)
            labels.append(b["labels"] + [-100] * k)
            attn.append([1] * len(b["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2816)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--max-bs", type=int, default=48)
    ap.add_argument("--liger", type=int, default=1)
    args = ap.parse_args()

    print("template sha256:", common_fmt.template_sha256())
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    tok = get_tokenizer(args.parent)
    feats = encode_rows(rows, tok, args.max_seq_len)
    random.Random(args.seed).shuffle(feats)

    if args.dry_run:
        ex = feats[0]
        txt = tok.decode(ex["input_ids"])
        print("=" * 30, "rendered example", "=" * 30)
        print(txt[:1500], "\n...\n", txt[-800:])
        print("=" * 30, "loss span", "=" * 30)
        print(tok.decode([t for t in ex["labels"] if t != -100])[-400:])
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    if hasattr(model, "vision_tower"):
        for p in model.vision_tower.parameters():
            p.requires_grad = False
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = False
        print("froze vision tower + projector")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=4,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=bool(args.liger),
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    class DS(Dataset):
        def __init__(self, feats):
            self.feats = feats

        def __len__(self):
            return len(self.feats)

        def __getitem__(self, i):
            return self.feats[i]

    lengths = [len(f["input_ids"]) for f in feats]
    batches = token_budget_batches(lengths, args.token_budget, args.max_bs, args.seed)
    print(f"{len(batches)} micro-batches, "
          f"mean size {len(feats)/len(batches):.1f}, "
          f"max padded tokens {max(len(b)*max(lengths[i] for i in b) for b in batches)}")

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader
            return DataLoader(
                DS(feats),
                batch_sampler=BatchList(batches, args.epochs),
                collate_fn=Collator(tok.pad_token_id or 0),
                num_workers=2,
            )

    class BatchList:
        """Epoch-aware batch sampler: reshuffles batch order each epoch."""

        def __init__(self, batches, epochs):
            self.batches = batches
            self.epochs = epochs
            self.epoch = 0

        def __len__(self):
            return len(self.batches)

        def __iter__(self):
            b = list(self.batches)
            random.Random(1000 + self.epoch).shuffle(b)
            self.epoch += 1
            yield from b

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=DS(feats),
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()
    final = os.path.join(args.output_dir, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # vLLM loads gemma3 as a multimodal model; keep the processor files the
    # base snapshot shipped (pitfall: final_model_not_loadable)
    import shutil
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.parent, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, fn)):
            shutil.copy(src, os.path.join(final, fn))
    print("saved", final)


if __name__ == "__main__":
    main()
