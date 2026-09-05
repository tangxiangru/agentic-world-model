"""Full-parameter SFT of gemma-3-4b-pt on pre-rendered prompt/completion pairs.

The jsonl rows already contain the exact strings the grader renders
(scripts/build_data.py -> scripts/fmt.py), so this script only tokenizes with
add_special_tokens=False and masks the prompt out of the loss.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_drop = 0
        lens = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_seq_len:
                    n_drop += 1
                    continue
                self.rows.append((p, c))
                lens.append(len(p) + len(c))
        lens.sort()
        self.lengths = [len(p) + len(c) for p, c in self.rows]
        print(
            f"[data] kept {len(self.rows)} rows, dropped {n_drop} over max_seq_len={max_seq_len} "
            f"({n_drop / max(1, n_drop + len(self.rows)):.2%}); "
            f"len p50={lens[len(lens) // 2]} p99={lens[int(0.99 * (len(lens) - 1))]} max={lens[-1]}",
            flush=True,
        )
        self.total_tokens = sum(self.lengths)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        return {"input_ids": p + c, "labels": [-100] * len(p) + list(c)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = m - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def save_full(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(SNAPSHOT):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = os.path.join(SNAPSHOT, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(SNAPSHOT).save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(SNAPSHOT).save_pretrained(out_dir)
    # greedy decoding: vLLM reads temperature/top_p/top_k out of generation_config.json
    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "cache_implementation": "hybrid",
    }
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)
    print(f"[save] wrote {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.limit)
    print(f"[data] total tokens {ds.total_tokens / 1e6:.1f}M", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    for n, p in model.named_parameters():
        if n.startswith("model.vision_tower") or n.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params {n_train / 1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )
    # group_by_length needs the lengths without touching __getitem__
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    t0 = time.time()
    out = trainer.train()
    dt = time.time() - t0
    print(f"[train] {dt / 60:.1f} min, {out.metrics}", flush=True)

    save_full(model, os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
