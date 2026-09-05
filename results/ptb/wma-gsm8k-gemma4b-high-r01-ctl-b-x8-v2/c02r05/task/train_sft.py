"""Completion-only SFT for gemma-3-4b-pt on grader-formatted math solutions.

Prompts are rendered through prompt_fmt (same jinja file the grader uses), the
prompt tokens are masked out of the loss, and every target ends with
<end_of_turn> (token 106), which is in the base generation_config's
eos_token_id list, so vLLM stops there at grading time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from prompt_fmt import FEWSHOT_SYSTEM, render_prompt, END_OF_TURN

HERE = os.path.dirname(os.path.abspath(__file__))


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, fewshot_frac, seed, limit=None):
        self.tok = tok
        self.max_seq_len = max_seq_len
        rows = [json.loads(l) for l in open(path)]
        if limit:
            rows = rows[:limit]
        rng = random.Random(seed)
        self.examples = []
        n_trunc = 0
        for r in rows:
            system = FEWSHOT_SYSTEM if rng.random() < fewshot_frac else None
            prompt = render_prompt(r["question"], system)
            completion = r.get("completion") or (r["solution"].strip() + END_OF_TURN)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(completion, add_special_tokens=False)["input_ids"]
            ids = p_ids + c_ids
            if len(ids) > max_seq_len:
                n_trunc += 1
                continue
            labels = [-100] * len(p_ids) + c_ids
            self.examples.append((ids, labels))
        self.n_trunc = n_trunc
        print(f"[data] {len(self.examples)} rows kept, {n_trunc} dropped for length "
              f"(> {max_seq_len})", flush=True)
        lens = sorted(len(e[0]) for e in self.examples)
        print(f"[data] token len p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} "
              f"max={lens[-1]}", flush=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labels = self.examples[i]
        return {"input_ids": ids, "labels": labels}


def collate(features, pad_id):
    n = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        k = n - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * k)
        labels.append(f["labels"] + [-100] * k)
        attn.append([1] * len(f["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--fewshot-frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--liger", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTRows(args.data, tok, args.max_seq_len, args.fewshot_frac, args.seed,
                 args.limit)

    if args.dry_run:
        ids, labels = ds.examples[0]
        print("=== decoded row 0 ===")
        print(tok.decode(ids))
        print("=== supervised part ===")
        print(tok.decode([i for i, l in zip(ids, labels) if l != -100]))
        return

    if args.liger:
        # fused linear cross-entropy: gemma-3's 262k vocab otherwise materialises
        # a batch x seq x 262144 fp32 logit tensor and OOMs the H100
        from liger_kernel.transformers import monkey_patch as _mp

        _mp.apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("[model] liger fused-linear-CE patch applied", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        for p in model.model.vision_tower.parameters():
            p.requires_grad = False
        for p in model.model.multi_modal_projector.parameters():
            p.requires_grad = False
        print("[model] vision tower + projector frozen", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        group_by_length=True,
        length_column_name=None,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda f: collate(f, tok.pad_token_id or 0),
    )
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # gemma-3 is a multimodal checkpoint: vLLM refuses to load the directory
    # without the processor files, which Trainer.save_model does not write
    # (pitfall: final_model_not_loadable)
    import shutil as _sh
    for extra in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, extra)
        if os.path.exists(src):
            _sh.copy2(src, os.path.join(final, extra))
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
