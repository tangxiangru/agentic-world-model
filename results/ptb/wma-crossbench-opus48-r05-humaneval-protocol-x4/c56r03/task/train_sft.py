#!/usr/bin/env python3
"""LoRA SFT for gemma-3-4b-pt on Python coding data, rendered with the grader's
gemma3.jinja template. Completion-only loss (prompt masked). Targets end in
<end_of_turn> (token 106), which is in the model's eos list so vLLM stops there.
Saves a merged (adapters folded) model + tokenizer to --out for vLLM loading.
"""
import argparse, json, os, random
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--template", default="templates/gemma3.jinja")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=0)
    return p.parse_args()


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main():
    a = parse()
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.chat_template = open(a.template).read()

    raw = [json.loads(l) for l in open(a.data)]
    if a.limit:
        raw = raw[:a.limit]

    rows, dropped, lens = [], 0, []
    for r in raw:
        # prompt already contains <bos>; do not add special tokens again.
        prompt_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        comp_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + comp_ids
        if len(full_ids) > a.max_seq_len:
            dropped += 1; continue
        labels = [-100] * len(prompt_ids) + comp_ids
        rows.append({"input_ids": full_ids, "labels": labels})
        lens.append(len(full_ids))
    lens.sort()
    p50 = lens[len(lens)//2]; p95 = lens[int(len(lens)*0.95)]; mx = lens[-1]
    print(f"kept {len(rows)} rows, dropped {dropped} (>{a.max_seq_len} or prefix mismatch); "
          f"len p50={p50} p95={p95} max={mx}")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=a.out + "_trainer",
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if a.save_steps else "no",
        save_steps=a.save_steps if a.save_steps else 1000000,
        report_to=[],
        seed=a.seed,
        dataloader_num_workers=4,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=SFTDataset(rows),
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    print("merging LoRA and saving merged model ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    print("saved merged model to", a.out)


if __name__ == "__main__":
    main()
