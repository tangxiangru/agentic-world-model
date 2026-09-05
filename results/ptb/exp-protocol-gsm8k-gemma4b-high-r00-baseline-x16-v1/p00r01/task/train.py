#!/usr/bin/env python3
"""LoRA SFT of gemma-3-4b-pt on GSM8K train, completion-only loss.

Renders with templates/gemma3.jinja (the grader's template, byte-for-byte),
ends every target with <end_of_turn> (the grader's stop token), and masks
loss to the completion span only.
"""
import json
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
MAXLEN = 768


class SFTData(Dataset):
    def __init__(self, path, tok):
        self.rows = []
        n_trunc = 0
        for line in open(path):
            r = json.loads(line)
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(r["completion"] + "<end_of_turn>\n", add_special_tokens=False)["input_ids"]
            ids = p_ids + c_ids
            if len(ids) > MAXLEN:
                n_trunc += 1
                continue
            labels = [-100] * len(p_ids) + c_ids[:]
            self.rows.append({"input_ids": ids, "labels": labels})
        print(f"[data] {len(self.rows)} rows kept, {n_trunc} dropped for len>{MAXLEN}")
        assert n_trunc / max(1, len(self.rows)) < 0.02, "too many rows truncated"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        d = m - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * d)
        labels.append(b["labels"] + [-100] * d)
        attn.append([1] * len(b["input_ids"]) + [0] * d)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_data.jsonl")
    ap.add_argument("--out", default="lora_out")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()

    ds = SFTData(args.data, tok)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        SNAP, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False

    # LoRA on the language model only -- never the siglip vision tower.
    lcfg = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id or 0),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved adapter ->", args.out)


if __name__ == "__main__":
    main()
