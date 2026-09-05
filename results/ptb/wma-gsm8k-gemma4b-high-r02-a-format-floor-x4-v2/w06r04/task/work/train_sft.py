"""Full fine-tune of google/gemma-3-4b-pt on pre-rendered prompt/completion rows.

The data file already contains the prompt rendered with the grader's own
templates/gemma3.jinja and a completion that ends in "ANSWER: n<end_of_turn>",
so this script does no formatting of its own: it tokenizes both halves,
masks the prompt out of the loss, and trains.

Rows whose tokenized length exceeds --max-seq-len are DROPPED, never truncated:
under completion-only loss a truncated row silently carries no answer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

END_OF_TURN = 106
EOS = 1


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Batches with a fixed token budget: rows_in_batch * longest_row <= budget.

    The vocabulary is 262k, so the logits tensor (batch x seq x 262144, upcast to
    fp32 inside the loss) is what actually decides peak memory - a fixed row count
    OOMs the moment a batch of long rows comes up (it did, at bs=8 x 2048).
    Budgeting tokens instead makes peak memory flat across the run.
    """

    def __init__(self, lengths, budget, seed=0):
        self.budget = budget
        self.seed = seed
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches, cur, curmax = [], [], 0
        for i in order:
            m = max(curmax, lengths[i])
            if cur and m * (len(cur) + 1) > budget:
                self.batches.append(cur)
                cur, curmax = [i], lengths[i]
            else:
                cur, curmax = cur + [i], m
        if cur:
            self.batches.append(cur)
        self.epoch = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        import random

        r = random.Random(self.seed + self.epoch)
        self.epoch += 1
        b = list(self.batches)
        r.shuffle(b)
        return iter(b)


class PackedRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id: int):
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


def build_rows(path, tok, max_seq_len, limit=None):
    rows, dropped = [], 0
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"]
            if not c or c[-1] != END_OF_TURN:
                raise ValueError(f"row {i}: completion does not end in <end_of_turn>")
            c = c + [EOS]  # so the run is safe even if generation_config loses eos 106
            if len(p) + len(c) > max_seq_len:
                dropped += 1
                continue
            rows.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    return rows, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=16384,
                    help="max tokens per micro-batch (rows * longest row)")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--liger", action="store_true")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--logging-steps", type=int, default=10)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows, dropped = build_rows(args.data, tok, args.max_seq_len, args.limit)
    ntok = sum(len(r["input_ids"]) for r in rows)
    nsup = sum(sum(1 for x in r["labels"] if x != -100) for r in rows)
    print(f"rows={len(rows)} dropped_too_long={dropped} "
          f"({dropped / max(1, len(rows) + dropped):.4%}) tokens={ntok} supervised={nsup}",
          flush=True)

    cfg = AutoConfig.from_pretrained(args.model)
    arch = cfg.architectures[0]
    import transformers

    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger: fused linear cross-entropy enabled for gemma3", flush=True)

    cls = getattr(transformers, arch)
    try:
        model = cls.from_pretrained(args.model, dtype=torch.bfloat16,
                                    attn_implementation="flash_attention_2")
    except Exception as e:  # flash-attn not installed / unsupported
        print(f"flash_attention_2 unavailable ({type(e).__name__}), using sdpa", flush=True)
        model = cls.from_pretrained(args.model, dtype=torch.bfloat16,
                                    attn_implementation="sdpa")

    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen={frozen / 1e9:.3f}B trainable={trainable / 1e9:.3f}B", flush=True)

    model.config.use_cache = False
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    lengths = [len(r["input_ids"]) for r in rows]
    sampler = TokenBudgetBatches(lengths, args.token_budget, seed=args.seed)
    print(f"micro-batches/epoch={len(sampler)} "
          f"median_rows/batch={sorted(len(b) for b in sampler.batches)[len(sampler) // 2]} "
          f"optimizer_steps/epoch={len(sampler) // args.grad_accum}", flush=True)

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self):
            from torch.utils.data import DataLoader

            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=targs.dataloader_num_workers,
                pin_memory=True,
            )

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=PackedRows(rows),
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    print("saving to", args.out, flush=True)
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(args.out)
    except Exception as e:
        print("processor save skipped:", type(e).__name__, e, flush=True)
    # Decode contract: ship the parent's generation_config verbatim so that a
    # candidate/comparator delta can never hide a decoder change.
    shutil.copyfile(os.path.join(args.model, "generation_config.json"),
                    os.path.join(args.out, "generation_config.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
