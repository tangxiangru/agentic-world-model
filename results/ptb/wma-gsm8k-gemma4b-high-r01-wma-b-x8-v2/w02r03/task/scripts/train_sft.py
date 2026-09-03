"""Full fine-tune of google/gemma-3-4b-pt on prompt/completion jsonl rows.

- vision tower + multimodal projector are frozen (text-only task)
- completion-only loss (prompt tokens masked to -100)
- rows longer than --max-seq-len are DROPPED, not truncated, so no row can
  silently carry zero loss tokens (pitfall: seq_len_truncation)
"""
import argparse
import json
import math
import os
import random
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, "/home/ben/task/scripts")
import render  # noqa: E402


class Rows(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None, report=None):
        self.ex = []
        n_drop = 0
        lens = []
        with open(path) as fh:
            for i, line in enumerate(fh):
                if limit and len(self.ex) >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False).input_ids
                c = tok(r["completion"], add_special_tokens=False).input_ids
                if len(p) + len(c) > max_seq_len:
                    n_drop += 1
                    continue
                lens.append(len(p) + len(c))
                self.ex.append((p, c))
        lens.sort()
        self.stats = {
            "kept": len(self.ex),
            "dropped": n_drop,
            "drop_frac": n_drop / max(1, n_drop + len(self.ex)),
            "p50": lens[len(lens) // 2] if lens else 0,
            "p99": lens[int(len(lens) * 0.99)] if lens else 0,
            "max": lens[-1] if lens else 0,
        }
        print("dataset stats:", self.stats, flush=True)
        if report:
            with open(report, "w") as fh:
                json.dump(self.stats, fh, indent=2)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        p, c = self.ex[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Batches of variable size under a fixed token budget.

    group_by_length with a fixed sample count OOMs on this task: one batch of
    eight 1700-token 10-shot rows materialises a 8*1700*262144 logits tensor
    plus its fp32 copy inside the loss. Budgeting tokens instead of rows keeps
    that tensor constant-size and cuts padding waste at the same time.
    """

    def __init__(self, lengths, budget, max_bs, seed, mega=4096):
        self.batches = []
        rng = random.Random(seed)
        idx = list(range(len(lengths)))
        rng.shuffle(idx)
        for s in range(0, len(idx), mega):
            chunk = sorted(idx[s : s + mega], key=lambda i: -lengths[i])
            cur, curmax = [], 0
            for i in chunk:
                m = max(curmax, lengths[i])
                if cur and ((len(cur) + 1) * m > budget or len(cur) + 1 > max_bs):
                    self.batches.append(cur)
                    cur, curmax = [i], lengths[i]
                else:
                    cur.append(i)
                    curmax = m
            if cur:
                self.batches.append(cur)
        rng.shuffle(self.batches)
        sizes = [len(b) for b in self.batches]
        print(
            f"token-budget batches: {len(self.batches)} "
            f"(mean bs {sum(sizes)/len(sizes):.1f}, min {min(sizes)}, max {max(sizes)})",
            flush=True,
        )

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = m - len(b["input_ids"])
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
    ap.add_argument("--model", default=render.MODEL_PATH)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--max-bs", type=int, default=48)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    proc = AutoProcessor.from_pretrained(render.MODEL_PATH)
    tok = proc.tokenizer
    tok.chat_template = render.template_text()

    ds = Rows(
        args.data,
        tok,
        args.max_seq_len,
        limit=args.limit,
        report=os.path.join(os.path.dirname(args.out) or ".", "dataset_stats.json"),
    )
    if ds.stats["drop_frac"] > 0.02:
        print(
            f"WARNING drop_frac={ds.stats['drop_frac']:.4f} > 2%; raise --max-seq-len",
            flush=True,
        )
    if args.dry_run:
        p, c = ds.ex[0]
        print("--- sample prompt ---")
        print(tok.decode(p))
        print("--- sample target ---")
        print(tok.decode(c))
        print("last label token:", c[-1], repr(tok.decode([c[-1]])))
        assert c[-1] == tok.convert_tokens_to_ids(render.STOP_TOKEN), "stop token!"
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M, trainable {n_train/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        adam_beta2=0.95,
    )

    lengths = [len(p) + len(c) for p, c in ds.ex]
    sampler = TokenBudgetBatches(lengths, args.token_budget, args.max_bs, args.seed)

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader

            dl = DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=targs.dataloader_num_workers,
                pin_memory=True,
            )
            return self.accelerator.prepare(dl)

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    proc.save_pretrained(args.out)
    # keep the grader's generation settings reachable
    import shutil

    for f in ["generation_config.json", "preprocessor_config.json"]:
        src = os.path.join(render.MODEL_PATH, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
