"""Completion-only SFT for gemma-3-4b-pt on prompt/completion jsonl.

Rows are pre-rendered by scripts/build_sft_data.py through scripts/fmt.py, i.e. the
prompt string already carries <bos><start_of_turn>user ... <start_of_turn>model\n and
the completion already ends with <end_of_turn>.  Nothing here re-applies a template.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

IGNORE = -100


class PC(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        self.ex = []
        n_trunc = n_drop = 0
        lens = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False).input_ids
                c = tok(r["completion"], add_special_tokens=False).input_ids
                lens.append(len(p) + len(c))
                if len(p) + len(c) > max_len:
                    n_trunc += 1
                    if len(p) >= max_len - 16:
                        n_drop += 1
                        continue
                    # keep the stop token if we must cut the tail
                    c = c[: max_len - len(p) - 1] + [c[-1]]
                ids = p + c
                labels = [IGNORE] * len(p) + c
                self.ex.append((ids, labels))
        lens.sort()
        self.stats = {
            "n_rows": len(lens),
            "p50": lens[len(lens) // 2],
            "p99": lens[int(len(lens) * 0.99)],
            "max": lens[-1],
            "over_max_len": n_trunc,
            "dropped": n_drop,
            "trunc_frac": round(n_trunc / max(1, len(lens)), 5),
        }

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        ids, labels = self.ex[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        ii, ll, am = [], [], []
        for f in feats:
            k = m - len(f["input_ids"])
            ii.append(f["input_ids"] + [self.pad_id] * k)
            ll.append(f["labels"] + [IGNORE] * k)
            am.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ii),
            "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(am),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=fmt.BASE_MODEL)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--liger", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = fmt.template_text()

    ds = PC(args.data, tok, args.max_len, args.limit)
    print("DATA STATS", json.dumps(ds.stats), flush=True)
    ids, labels = ds.ex[0]
    print("EXAMPLE last ids", ids[-4:], "labels tail", labels[-4:],
          "stop_ok", ids[-1] == tok.convert_tokens_to_ids(fmt.STOP_TOKEN), flush=True)
    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    if args.liger:
        # gemma-3's 262k vocab makes the fp32 cross-entropy logits the memory wall;
        # liger's fused linear-CE never materialises them.
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        _apply_liger_kernel_to_instance(model=model)
        print("liger kernel applied", flush=True)
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen vision params: {n_frozen/1e6:.1f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id)
    )
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # the grader loads with vLLM from a fresh process; keep the processor files too
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # noqa: BLE001
        print("processor save skipped:", e)
    print("SAVED", final, flush=True)


if __name__ == "__main__":
    main()
