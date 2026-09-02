#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style data.

Reads a jsonl of {"prompt": str, "completion": str} where both strings are already
rendered exactly the way templates/gemma3.jinja renders the graded turn (prompt ends
with "<start_of_turn>model\\n", completion ends with "<end_of_turn>\\n"); see
scripts/verify_data.py for the byte-for-byte check.  Loss is on completion tokens only.

Batches are built to a token budget rather than a fixed row count, because the corpus
mixes ~350-token 0-shot rows with ~2400-token rows carrying the eval's 10-shot prefix.
"""
import argparse, json, os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import (AutoTokenizer, AutoModelForImageTextToText, Trainer,
                          TrainingArguments, set_seed)

SNAP = os.environ.get("PTB_BASE_MODEL_SNAPSHOT")


class PromptCompletionDataset(Dataset):
    def __init__(self, rows, tok, max_len, nproc=8):
        self.ex = []
        self.n_trunc = 0
        prompts = [r[0] for r in rows]
        comps = [r[1] for r in rows]
        pids = tok(prompts, add_special_tokens=True).input_ids      # adds <bos>
        cids = tok(comps, add_special_tokens=False).input_ids
        for pi, ci in zip(pids, cids):
            if len(pi) + len(ci) > max_len:
                self.n_trunc += 1
                continue                                            # never truncate a target
            self.ex.append((pi + ci, [-100] * len(pi) + ci))
        self.lengths = [len(e[0]) for e in self.ex]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, labels = self.ex[i]
        return {"input_ids": ids, "labels": labels}


class TokenBudgetBatchSampler(Sampler):
    """Length-bucketed batches capped at `max_tokens` padded tokens each."""

    def __init__(self, lengths, max_tokens, seed=0, megabatch=4096):
        self.batches = []
        rng = random.Random(seed)
        idx = list(range(len(lengths)))
        rng.shuffle(idx)
        for i in range(0, len(idx), megabatch):
            chunk = sorted(idx[i:i + megabatch], key=lambda j: lengths[j])
            cur, curmax = [], 0
            for j in chunk:
                nm = max(curmax, lengths[j])
                if cur and nm * (len(cur) + 1) > max_tokens:
                    self.batches.append(cur)
                    cur, curmax = [j], lengths[j]
                else:
                    cur.append(j)
                    curmax = nm
            if cur:
                self.batches.append(cur)
        rng.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        m = (m + 7) // 8 * 8
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = m - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


class TokenBudgetTrainer(Trainer):
    batch_sampler = None

    def get_train_dataloader(self):
        return self.accelerator.prepare(DataLoader(
            self.train_dataset, batch_sampler=self.batch_sampler,
            collate_fn=self.data_collator, num_workers=4, pin_memory=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--max-tokens-per-batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-epochs", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--no-liger", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = []
    with open(args.data) as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["prompt"], d["completion"]))
    if args.limit:
        rows = rows[: args.limit]
    ds = PromptCompletionDataset(rows, tok, args.max_seq_len)
    lens = np.array(ds.lengths)
    print(f"dataset: {len(ds)} rows kept, {ds.n_trunc} dropped over max_seq_len={args.max_seq_len} "
          f"({ds.n_trunc/max(1,len(rows)):.3%})", flush=True)
    print(f"token len p50={np.percentile(lens,50):.0f} p95={np.percentile(lens,95):.0f} "
          f"max={lens.max()} total={lens.sum()/1e6:.1f}M", flush=True)

    sampler = TokenBudgetBatchSampler(ds.lengths, args.max_tokens_per_batch, seed=args.seed)
    print(f"{len(sampler)} micro-batches/epoch -> {len(sampler)//args.accum} optimizer steps/epoch",
          flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():           # text-only task: freeze SigLIP
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M, trainable {n_train/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,          # unused: batch_sampler drives the loader
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=10,
        save_strategy=("epoch" if args.save_epochs else ("steps" if args.save_steps else "no")),
        save_steps=(args.save_steps or 500),
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        report_to=[],
        seed=args.seed,
        save_safetensors=True,
        use_liger_kernel=(not args.no_liger),
        accelerator_config={"dispatch_batches": False},
    )
    trainer = TokenBudgetTrainer(model=model, args=targs, train_dataset=ds,
                                 data_collator=Collator(tok.pad_token_id))
    trainer.batch_sampler = sampler
    import transformers.models.gemma3.modeling_gemma3 as mg3
    print("liger patched:", "liger" in str(type(trainer.model.model.language_model.norm)).lower() or "liger" in mg3.Gemma3MLP.__name__.lower(), type(trainer.model.model.language_model.norm), flush=True)
    trainer.train()
    print(f"peak cuda mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
