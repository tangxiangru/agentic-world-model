import json, os, argparse, random
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments)
from torch.utils.data import Dataset

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train_full.jsonl")
    p.add_argument("--out", default="sft_model")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--maxlen", type=int, default=1280)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--warmup", type=float, default=0.03)
    return p.parse_args()

class FCDataset(Dataset):
    def __init__(self, path, tok, maxlen, limit=0):
        self.examples = []
        n_skip = 0
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        if limit:
            rows = rows[:limit]
        for r in rows:
            prompt_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            comp_text = r["text"][len(r["prompt"]):]
            comp_ids = tok(comp_text, add_special_tokens=False)["input_ids"]
            ids = prompt_ids + comp_ids
            if len(ids) > maxlen:
                n_skip += 1
                continue
            labels = [-100] * len(prompt_ids) + list(comp_ids)
            self.examples.append({"input_ids": ids, "labels": labels})
        print(f"dataset: {len(self.examples)} examples, skipped {n_skip} (too long)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

def main():
    args = parse()
    tok = AutoTokenizer.from_pretrained(SNAP)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    ds = FCDataset(args.data, tok, args.maxlen, args.limit)

    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to=[],
        dataloader_num_workers=4,
        seed=0,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(pad_id))
    trainer.train()
    # restore cache flags so generation_config validation passes on save
    model.config.use_cache = True
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved to", args.out)

if __name__ == "__main__":
    main()
