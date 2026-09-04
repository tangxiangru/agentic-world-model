import os, json, argparse
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, AutoModelForImageTextToText)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gsm8k_sft.jsonl")
    p.add_argument("--out", default="runs/sft1")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--maxlen", type=int, default=1024)
    p.add_argument("--warmup", type=float, default=0.03)
    return p.parse_args()

def main():
    args = get_args()
    tok = AutoTokenizer.from_pretrained(SNAP)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=args.data, split="train")

    def encode(ex):
        msgs = ex["messages"]
        prompt_text = tok.apply_chat_template(msgs[:1], add_generation_prompt=True, tokenize=False)
        full_text = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:args.maxlen]
        labels = list(full_ids)
        plen = min(len(prompt_ids), len(full_ids))
        for i in range(plen):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels}

    ds = ds.map(encode, remove_columns=ds.column_names, num_proc=8)
    # drop examples where everything is masked
    ds = ds.filter(lambda e: any(l != -100 for l in e["labels"]))
    print("dataset size", len(ds))
    lens = [len(e) for e in ds["input_ids"][:2000]]
    print("sample token lens min/max/avg", min(lens), max(lens), sum(lens)/len(lens))

    class Collator:
        def __init__(self, tok):
            self.pad = tok.pad_token_id
        def __call__(self, feats):
            maxlen = max(len(f["input_ids"]) for f in feats)
            input_ids, labels, attn = [], [], []
            for f in feats:
                ids = f["input_ids"]; lab = f["labels"]
                pad = maxlen - len(ids)
                input_ids.append(ids + [self.pad]*pad)
                labels.append(lab + [-100]*pad)
                attn.append([1]*len(ids) + [0]*pad)
            return {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn),
            }

    model = AutoModelForImageTextToText.from_pretrained(
        SNAP, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    # freeze vision tower + projector -> train language model only
    frozen = 0; trained = 0
    for n, p in model.named_parameters():
        if ("vision_tower" in n) or ("multi_modal_projector" in n):
            p.requires_grad_(False); frozen += p.numel()
        else:
            trained += p.numel()
    print(f"trainable params: {trained/1e9:.2f}B, frozen: {frozen/1e9:.2f}B")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=4,
        save_only_model=True,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to=[],
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(tok),
                      processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved to", args.out)

if __name__ == "__main__":
    main()
