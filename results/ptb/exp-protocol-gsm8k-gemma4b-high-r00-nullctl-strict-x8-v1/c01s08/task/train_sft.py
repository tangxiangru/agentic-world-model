#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on math data in the inspect gsm8k prompt format."""
from __future__ import annotations
import argparse, json, math, os, random
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, set_seed)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        self.ex = []
        pre = tok("<bos><start_of_turn>user\n", add_special_tokens=False)["input_ids"]
        mid = tok("<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)["input_ids"]
        end = tok("<end_of_turn>", add_special_tokens=False)["input_ids"]
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
        if limit:
            rows = rows[:limit]
        prompts = tok([r["prompt"] for r in rows], add_special_tokens=False)["input_ids"]
        comps = tok([r["completion"] for r in rows], add_special_tokens=False)["input_ids"]
        n_drop = 0
        for p, c in zip(prompts, comps):
            pids = pre + p + mid
            cids = c + end
            if len(pids) + len(cids) > max_len:
                n_drop += 1
                continue
            self.ex.append((pids, cids))
        print(f"loaded {len(self.ex)} examples, dropped {n_drop} over {max_len} tokens")
        self.lengths = [len(a) + len(b) for a, b in self.ex]
        print("total tokens:", sum(self.lengths))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        p, c = self.ex[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        m = ((m + 63) // 64) * 64
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            pad = m - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class TokenBatchSampler(torch.utils.data.Sampler):
    """Groups similar-length examples into batches with a fixed padded-token budget."""

    def __init__(self, lengths, budget, seed=0):
        self.lengths = lengths
        self.budget = budget
        self.seed = seed
        self.epoch = 0
        self.batches = self._make(0)

    def _make(self, epoch):
        rng = random.Random(self.seed + epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        mega = 4096
        order = []
        for i in range(0, len(idx), mega):
            chunk = idx[i:i + mega]
            chunk.sort(key=lambda j: self.lengths[j])
            order += chunk
        batches, cur, curmax = [], [], 0
        for j in order:
            l = ((self.lengths[j] + 63) // 64) * 64
            newmax = max(curmax, l)
            if cur and newmax * (len(cur) + 1) > self.budget:
                batches.append(cur)
                cur, curmax = [j], l
            else:
                cur.append(j)
                curmax = newmax
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self.epoch = epoch
            self.batches = self._make(epoch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class TokenBatchTrainer(Trainer):
    token_batch_sampler = None

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader
        dl = DataLoader(self.train_dataset,
                        batch_sampler=self.token_batch_sampler,
                        collate_fn=self.data_collator,
                        num_workers=self.args.dataloader_num_workers,
                        pin_memory=True)
        return self.accelerator.prepare(dl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--out", default="ckpt/sft1")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-epochs", type=float, default=1.0)
    ap.add_argument("--fp32-master", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--token-budget", type=int, default=20480)
    args = ap.parse_args()
    set_seed(args.seed)

    from liger_kernel.transformers import apply_liger_kernel_to_gemma3
    apply_liger_kernel_to_gemma3()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTData(args.data, tok, args.max_len, args.limit)

    dtype = torch.float32 if args.fp32_master else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=dtype, attn_implementation="flash_attention_2")
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        do_sample=True, top_k=64, top_p=0.95, cache_implementation="hybrid")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    # freeze the vision stack -- we only ever feed text
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print("frozen params:", n_frozen / 1e6, "M ; trainable:",
          sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9, "B")

    sampler = TokenBatchSampler(ds.lengths, args.token_budget, seed=args.seed)
    print("micro-batches per epoch:", len(sampler),
          "mean batch size:", len(ds) / len(sampler))
    steps_per_epoch = math.ceil(len(sampler) / args.accum)
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=int(steps_per_epoch * args.save_epochs),
        save_total_limit=4,
        gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        save_safetensors=True,
        seed=args.seed,
    )
    TokenBatchTrainer.token_batch_sampler = sampler
    trainer = TokenBatchTrainer(model=model, args=targs, train_dataset=ds,
                                data_collator=Collator(tok.pad_token_id))
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    print("DONE")


if __name__ == "__main__":
    main()
