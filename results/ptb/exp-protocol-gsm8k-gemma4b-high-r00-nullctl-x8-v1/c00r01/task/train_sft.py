#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, random, shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig,
                          Trainer, TrainingArguments)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class JsonlSFT(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        self.rows = []
        n_skip = 0
        paths = path.split(",")
        for pth in paths:
            with open(pth) as f:
                for line in f:
                    r = json.loads(line)
                    p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                    c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                    ids = [tok.bos_token_id] + p + c
                    if len(ids) > max_len:
                        n_skip += 1
                        continue
                    labels = [-100] * (1 + len(p)) + c
                    self.rows.append((ids, labels))
                    if limit and len(self.rows) >= limit:
                        break
        self.lengths = np.array([len(r[0]) for r in self.rows])
        print(f"loaded {len(self.rows)} rows (skipped {n_skip} too long), "
              f"total tokens {self.lengths.sum()/1e6:.1f}M", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


class TokenBudgetSampler(torch.utils.data.Sampler):
    """Length-grouped batches with a padded-token budget per batch."""

    def __init__(self, lengths, max_tokens, seed=0, mega=2048, max_bs=64):
        self.lengths, self.max_tokens = lengths, max_tokens
        self.seed, self.mega, self.max_bs = seed, mega, max_bs
        self.epoch = 0
        self._batches = self._build(0)

    def _build(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        idx = rng.permutation(len(self.lengths))
        batches = []
        for s in range(0, len(idx), self.mega):
            chunk = idx[s:s + self.mega]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            cur, curmax = [], 0
            for i in chunk:
                nm = max(curmax, int(self.lengths[i]))
                if cur and (nm * (len(cur) + 1) > self.max_tokens or len(cur) >= self.max_bs):
                    batches.append(cur)
                    cur, curmax = [int(i)], int(self.lengths[i])
                else:
                    cur.append(int(i)); curmax = nm
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, e):
        self.epoch = e
        self._batches = self._build(e)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        m = ((m + 7) // 8) * 8
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * n)
            labels.append(f["labels"] + [-100] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


class BudgetTrainer(Trainer):
    sampler = None

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_sampler=self.sampler,
                          collate_fn=self.data_collator, num_workers=4,
                          pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--out", default="runs/sft1")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = JsonlSFT(args.data, tok, args.max_len, args.limit)

    cfg = AutoConfig.from_pretrained(args.init)
    is_mm = cfg.architectures[0] == "Gemma3ForConditionalGeneration"
    if is_mm:
        from transformers import Gemma3ForConditionalGeneration as Cls
    else:
        Cls = AutoModelForCausalLM
    from liger_kernel.transformers import apply_liger_kernel_to_gemma3
    apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
    model = Cls.from_pretrained(args.init, dtype=torch.bfloat16,
                                attn_implementation=os.environ.get("ATTN", "eager"))
    model.config.use_cache = False
    if is_mm:
        for n, p in model.named_parameters():
            if "vision_tower" in n or "multi_modal_projector" in n:
                p.requires_grad = False
    print(f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B",
          flush=True)

    sampler = TokenBudgetSampler(ds.lengths, args.max_tokens, seed=0)
    steps_per_epoch = math.ceil(len(sampler) / args.accum)
    total_steps = int(steps_per_epoch * args.epochs)
    print(f"micro-batches/epoch {len(sampler)}, optimizer steps {total_steps}", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        max_steps=total_steps,
        per_device_train_batch_size=1,  # unused (batch_sampler)
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup,
        logging_steps=25,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        weight_decay=0.0,
        max_grad_norm=1.0,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=0,
    )
    tr = BudgetTrainer(model=model, args=targs, train_dataset=ds,
                       data_collator=Collator(tok.pad_token_id))
    tr.sampler = sampler
    tr.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    tr.save_model(final)
    tok.save_pretrained(final)
    for fn in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    print("saved to", final)


if __name__ == "__main__":
    main()
