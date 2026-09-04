#!/usr/bin/env python3
"""SFT for gemma-3-4b-pt on GSM8K-style data.
- Renders with the EXACT grader template (templates/gemma3.jinja).
- Completion-only loss (prompt tokens masked to -100).
- Supports full fine-tuning or LoRA (--mode full|lora).
"""
import argparse, json, os, math, shutil
import torch
from transformers import (AutoTokenizer, AutoModelForImageTextToText,
                          TrainingArguments, Trainer, TrainerCallback)
from datasets import load_dataset

def save_full(model, tok, base_model, out, mode):
    os.makedirs(out, exist_ok=True)
    m = model.merge_and_unload() if mode == "lora" else model
    m.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    for fn in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(base_model, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, fn))

class EpochSaver(TrainerCallback):
    def __init__(self, model, tok, base_model, out, mode):
        self.model, self.tok, self.base, self.out, self.mode = model, tok, base_model, out, mode
    def on_epoch_end(self, args, state, control, **kw):
        ep = round(state.epoch)
        d = os.path.join(self.out, f"epoch-{ep}")
        # merge_and_unload is destructive for lora; only snapshot for full FT
        if self.mode == "full":
            save_full(self.model, self.tok, self.base, d, self.mode)
            print(f"[epoch-save] saved {d}")

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="templates/gemma3.jinja")
    ap.add_argument("--mode", default="full", choices=["full","lora"])
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--save_steps", type=int, default=0)
    ap.add_argument("--save_each_epoch", action="store_true")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()

def build_dataset(args, tok):
    tmpl = open(args.template).read()
    raw = load_dataset("json", data_files=args.data, split="train")
    if args.max_examples:
        raw = raw.select(range(min(args.max_examples, len(raw))))

    def encode(ex):
        user = {"role":"user","content":ex["prompt"]}
        asst = {"role":"assistant","content":ex["completion"]}
        prompt_str = tok.apply_chat_template([user], chat_template=tmpl,
                        tokenize=False, add_generation_prompt=True)
        full_str = tok.apply_chat_template([user, asst], chat_template=tmpl,
                        tokenize=False, add_generation_prompt=False)
        prompt_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_str, add_special_tokens=False)["input_ids"]
        # longest common prefix -> mask boundary
        n = 0
        for a,b in zip(prompt_ids, full_ids):
            if a==b: n+=1
            else: break
        labels = [-100]*n + full_ids[n:]
        return {"input_ids": full_ids, "labels": labels, "length": len(full_ids)}

    ds = raw.map(encode, remove_columns=raw.column_names, desc="tokenize")
    lengths = ds["length"]
    lengths_sorted = sorted(lengths)
    p50 = lengths_sorted[len(lengths_sorted)//2]
    mx = max(lengths)
    trunc = sum(1 for l in lengths if l > args.max_seq_len)
    print(f"[data] n={len(ds)} p50_len={p50} max_len={mx} "
          f"trunc(>{args.max_seq_len})={trunc} ({100*trunc/len(ds):.2f}%)")
    # truncate to max_seq_len
    def trunc_fn(ex):
        ex["input_ids"] = ex["input_ids"][:args.max_seq_len]
        ex["labels"] = ex["labels"][:args.max_seq_len]
        return ex
    ds = ds.map(trunc_fn, desc="truncate")
    ds = ds.remove_columns(["length"])
    return ds

class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = f["input_ids"]; lab = f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id]*pad)
            labels.append(lab + [-100]*pad)
            attn.append([1]*len(ids) + [0]*pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

def main():
    args = parse()
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = build_dataset(args, tok)
    if args.dry_run:
        print("[dry_run] dataset built, exiting before model load")
        return

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                        target_modules=targets, task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    save_kwargs = {}
    if args.save_steps > 0:
        save_kwargs = dict(save_strategy="steps", save_steps=args.save_steps)
    else:
        save_kwargs = dict(save_strategy="no")

    targ = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        seed=args.seed,
        optim=args.optim,
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **save_kwargs,
    )
    callbacks = []
    if args.save_each_epoch and args.mode == "full":
        callbacks.append(EpochSaver(model, tok, args.model, args.out, args.mode))
    trainer = Trainer(model=model, args=targ, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id), callbacks=callbacks)
    trainer.train()

    # save final (merged, with tokenizer + multimodal processor configs)
    save_full(model, tok, args.model, args.out, args.mode)
    print(f"[done] saved to {args.out}")

if __name__ == "__main__":
    main()
