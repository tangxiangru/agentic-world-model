#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt.

Rows are pre-rendered (scripts/build_sft.py) with the grader's own template, so
this script only tokenizes, masks the prompt, and trains. Nothing is truncated:
rows longer than --max-seq-len are dropped and counted, because under
completion-only loss a truncated row silently carries zero loss tokens
(pitfall `seq_len_truncation`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
AUX_FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
]


class PackedRows(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.ex = []
        n_drop = 0
        n_seen = 0
        with open(path) as f:
            for line in f:
                if limit and n_seen >= limit:
                    break
                n_seen += 1
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False).input_ids
                c = tok(r["completion"], add_special_tokens=False).input_ids
                if len(p) + len(c) > max_seq_len:
                    n_drop += 1
                    continue
                self.ex.append((p, c))
        self.n_drop = n_drop
        self.n_seen = n_seen
        self.lengths = [len(p) + len(c) for p, c in self.ex]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        p, c = self.ex[i]
        ids = p + c
        labels = [-100] * len(p) + c
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ii, ll, am = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ii.append(b["input_ids"] + [pad_id] * k)
        ll.append(b["labels"] + [-100] * k)
        am.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(ii, dtype=torch.long),
        "labels": torch.tensor(ll, dtype=torch.long),
        "attention_mask": torch.tensor(am, dtype=torch.long),
    }


def save_full(model, tok, outdir, greedy=True):
    os.makedirs(outdir, exist_ok=True)
    model.save_pretrained(outdir, safe_serialization=True)
    tok.save_pretrained(outdir)
    for fn in AUX_FILES:
        src = os.path.join(BASE_SNAPSHOT, fn)
        dst = os.path.join(outdir, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
    # evaluate.py sends no sampling params, so vLLM decodes with whatever this
    # file says. vLLM only reads repetition_penalty/temperature/top_k/top_p/
    # min_p/max_new_tokens from generation_config (vllm/config/model.py
    # get_diff_sampling_param) -- do_sample is IGNORED, which cost exp-02
    # 18.7 points until exp-03 found it. temperature is the field that works.
    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": False,
    }
    if greedy:
        gc["temperature"] = 0.0
    else:
        gc.update({"do_sample": True, "temperature": 1.0, "top_k": 64, "top_p": 0.95})
    with open(os.path.join(outdir, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=BASE_SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT)
    ds = PackedRows(args.data, tok, args.max_seq_len, limit=args.limit)
    frac = ds.n_drop / max(1, ds.n_seen)
    print(f"rows kept {len(ds)} / seen {ds.n_seen}; dropped-for-length {ds.n_drop} ({frac:.3%})", flush=True)
    assert frac < 0.02, f"more than 2% of rows exceed max_seq_len ({frac:.2%})"

    # sanity: every kept row has at least one loss token and ends on <end_of_turn>
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    for i in range(0, len(ds), max(1, len(ds) // 200)):
        p, c = ds.ex[i]
        assert len(c) > 0, "row with zero loss tokens"
        assert eot in c[-3:], f"row {i} target does not end on <end_of_turn>: {c[-3:]}"
    print("checked: every sampled row has loss tokens and terminates on <end_of_turn>", flush=True)

    # gemma-3's vocab is 262k, so a bs*seq of 16k tokens materialises a 17 GB
    # fp32 logit tensor inside cross_entropy and OOMs an 80 GB H100. Liger's
    # fused linear cross-entropy never materialises the logits.
    if not args.no_liger:
        from liger_kernel.transformers import monkey_patch

        monkey_patch.apply_liger_kernel_to_gemma3(
            rope=True, cross_entropy=False, fused_linear_cross_entropy=True,
            rms_norm=True, geglu=True,
        )
        print("liger fused-linear-CE enabled", flush=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    if hasattr(model.model, "vision_tower"):
        for p_ in model.model.vision_tower.parameters():
            p_.requires_grad_(False)
        for p_ in model.model.multi_modal_projector.parameters():
            p_.requires_grad_(False)
    model.config.use_cache = False
    # A parent saved by save_full carries do_sample:false together with
    # temperature:0.0, and transformers' GenerationConfig.save_pretrained raises
    # ValueError on that pair -- which would kill the run at save time, after the
    # epoch is already paid for. Reset to a serialisable config here; save_full
    # overwrites the file with the greedy JSON afterwards anyway.
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid",
    )
    n_tr = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"trainable params: {n_tr/1e9:.3f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 10**9,
        save_total_limit=3,
        save_only_model=True,
        save_safetensors=True,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        report_to=[],
        seed=args.seed,
        max_steps=8 if args.smoke else -1,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    out = trainer.train()
    print("train result:", out, flush=True)
    final = os.path.join(args.out, "final")
    save_full(trainer.model, tok, final)
    print("saved", final, flush=True)
    with open(os.path.join(args.out, "train_summary.json"), "w") as f:
        json.dump(
            {
                "final_loss": out.training_loss,
                "steps": out.global_step,
                "rows": len(ds),
                "dropped": ds.n_drop,
                "args": vars(args),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
