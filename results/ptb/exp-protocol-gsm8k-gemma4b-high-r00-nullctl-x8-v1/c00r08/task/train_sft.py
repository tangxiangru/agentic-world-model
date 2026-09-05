"""Supervised fine-tuning of google/gemma-3-4b-pt for GSM8K, in the exact eval format."""
import argparse
import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from common import BASE_SNAPSHOT, fewshot_block, gsm8k_fewshots, load_chat_template, user_prompt

IGNORE = -100


def build_examples(rows, tok, args):
    """Render each row to (prompt_ids, completion_ids) with the eval chat format."""
    rng = random.Random(args.seed)

    # exact eval system prompt (10 shots, seed 42, shuffled) + a pool for random shots
    eval_shots = gsm8k_fewshots(10, seed=42, shuffle=True)
    eval_sys = "\n\n".join(eval_shots)
    shot_pool = gsm8k_fewshots(200, seed=7, shuffle=True)

    prompts, completions = [], []
    for r in rows:
        p = rng.random()
        if p < args.p_eval_shots:
            system = eval_sys
        elif p < args.p_eval_shots + args.p_rand_shots:
            k = rng.randint(1, 4)
            system = "\n\n".join(rng.sample(shot_pool, k))
        else:
            system = None
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_prompt(r["question"])})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        completions.append(r["solution"].strip() + "<end_of_turn>\n")
    return prompts, completions


class SFTDataset(Dataset):
    def __init__(self, prompts, completions, tok, max_len):
        self.items = []
        B = 1000
        for i in range(0, len(prompts), B):
            pe = tok(prompts[i : i + B], add_special_tokens=False)["input_ids"]
            ce = tok(completions[i : i + B], add_special_tokens=False)["input_ids"]
            for a, b in zip(pe, ce):
                if len(a) + len(b) > max_len:
                    continue
                self.items.append((a, b))
        self.lengths = [len(a) + len(b) for a, b in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, b = self.items[i]
        ids = a + b
        labels = [IGNORE] * len(a) + list(b)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            lab.append(f["labels"] + [IGNORE] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_pool.jsonl")
    ap.add_argument("--init", default=BASE_SNAPSHOT)
    ap.add_argument("--out", default="runs/sft_v1")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max-len", type=int, default=2600)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-eval-shots", type=float, default=0.06)
    ap.add_argument("--p-rand-shots", type=float, default=0.12)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT)
    tok.chat_template = load_chat_template()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"rows: {len(rows)}")

    prompts, completions = build_examples(rows, tok, args)
    ds = SFTDataset(prompts, completions, tok, args.max_len)
    tot = sum(ds.lengths)
    print(f"examples kept: {len(ds)} | total tokens: {tot/1e6:.1f}M | mean {tot/max(1,len(ds)):.0f}")

    from liger_kernel.transformers import apply_liger_kernel_to_gemma3

    apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False
    # keep the (unused at eval) vision stack frozen
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_steps=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=2,
        max_steps=args.max_steps,
        gradient_checkpointing=not args.no_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        optim=args.optim,
        report_to=[],
        seed=args.seed,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # ship the eval-time processor config alongside the weights
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE_SNAPSHOT, fn)
        if os.path.exists(src):
            import shutil

            shutil.copy(src, os.path.join(final, fn))
    print("saved to", final)


if __name__ == "__main__":
    main()
