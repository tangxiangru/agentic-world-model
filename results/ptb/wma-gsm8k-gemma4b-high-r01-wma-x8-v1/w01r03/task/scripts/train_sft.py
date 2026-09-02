#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on grader-formatted math solutions.

Rows are pre-tokenized here (not by TRL) so the exact string the model trains on
is the string the grader renders: prompt via templates/gemma3.jinja, completion
terminated by <end_of_turn>.  Loss is completion-only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_fmt import (  # noqa: E402
    BASE_SNAPSHOT,
    STOP_TOKEN,
    fewshot_system_message,
    gsm8k_train_exemplars,
    load_tokenizer,
    render_prompt,
    template_sha,
)


def make_prefix_sampler(mode: str, seed: int):
    """Return f(rng) -> system message or None.

    mode 'none'   : never a prefix (exp-02 behaviour)
    mode 'grader' : always the grader's exact 10-shot block
    mode 'mixed'  : half the rows get the grader's exact block, half get k
                    random GSM8K-train exemplars (k in {4,6,8,10}) in a random
                    order, so the model learns 'answer the last question and
                    stop' rather than memorising one prefix.
    """
    if mode == "none":
        return lambda rng: None
    grader = fewshot_system_message()
    if mode == "grader":
        return lambda rng: grader
    pool = gsm8k_train_exemplars()

    def sample(rng):
        if rng.random() < 0.5:
            return grader
        k = rng.choice([4, 6, 8, 10])
        return "\n\n".join(rng.sample(pool, k))

    return sample


def build_dataset(tok, path, max_seq_len, fewshot_frac, seed, limit=None, fewshot_mode="grader"):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    prefix_of = make_prefix_sampler(fewshot_mode if fewshot_frac > 0 else "none", seed)
    rng = random.Random(seed)

    feats, n_trunc, lens = [], 0, []
    for r in rows:
        sysmsg = prefix_of(rng) if rng.random() < fewshot_frac else None
        prompt = render_prompt(tok, r["problem"], system=sysmsg)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        completion = r["completion"]
        if not completion.endswith(STOP_TOKEN):
            completion += STOP_TOKEN
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        total = len(p_ids) + len(c_ids)
        lens.append(total)
        if total > max_seq_len:
            n_trunc += 1
            continue
        feats.append(
            {
                "input_ids": p_ids + c_ids,
                "labels": [-100] * len(p_ids) + c_ids,
                "length": total,
            }
        )
    lens.sort()
    print(
        json.dumps(
            {
                "rows_in": len(rows),
                "rows_kept": len(feats),
                "dropped_too_long": n_trunc,
                "drop_share": round(n_trunc / max(1, len(rows)), 4),
                "len_p50": lens[len(lens) // 2],
                "len_p99": lens[int(len(lens) * 0.99)],
                "len_max": lens[-1],
                "max_seq_len": max_seq_len,
            },
            indent=2,
        ),
        flush=True,
    )
    return feats


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=BASE_SNAPSHOT)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--fewshot-mode", default="mixed", choices=["none", "grader", "mixed"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--load-dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("template sha", template_sha(), flush=True)
    tok = load_tokenizer(BASE_SNAPSHOT)
    feats = build_dataset(
        tok, args.data, args.max_seq_len, args.fewshot_frac, args.seed, args.limit,
        args.fewshot_mode,
    )
    print("example decoded target tail:", repr(tok.decode(feats[0]["input_ids"][-12:])), flush=True)
    assert feats[0]["input_ids"][-1] == tok.convert_tokens_to_ids(STOP_TOKEN)
    if args.dry_run:
        return

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    ds = Dataset.from_list(feats)

    model = AutoModelForCausalLM.from_pretrained(
        args.parent,
        dtype=getattr(torch, args.load_dtype),
        attn_implementation=args.attn,
    )
    # text-only training: the vision stack is kept (so the checkpoint loads the
    # same way the base does) but never updated
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        optim=args.optim,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        dataloader_num_workers=4,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    out = trainer.train()
    print("train result", out, flush=True)

    final = os.path.join(args.output_dir, "final")
    # save in the base checkpoint's dtype so the dir loads exactly like the base
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    if hasattr(model.config, "text_config"):
        model.config.text_config.torch_dtype = "bfloat16"
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the processor/preprocessor files so the dir loads exactly like the base
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.parent, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, fn)):
            with open(src) as a, open(os.path.join(final, fn), "w") as b:
                b.write(a.read())
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
