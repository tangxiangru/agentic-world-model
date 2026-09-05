import argparse, json, os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration,
    Trainer, TrainingArguments,
)

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--save_steps", type=int, default=0)
    return p.parse_args()

args = parse()
tok = AutoTokenizer.from_pretrained(args.model)
print("bos:", repr(tok.bos_token), "eos:", repr(tok.eos_token), "pad:", repr(tok.pad_token))

BOS = tok.bos_token
def build_prompt(user_content):
    return f"{BOS}<start_of_turn>user\n{user_content.strip()}<end_of_turn>\n<start_of_turn>model\n"

ds = load_dataset("json", data_files=args.data, split="train")

def tokenize(ex):
    prompt = build_prompt(ex["prompt"])
    completion = ex["completion"].strip() + "<end_of_turn>\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    # preserve completion; truncate prompt from the left if needed
    if len(p_ids) + len(c_ids) > args.max_len:
        keep_p = max(0, args.max_len - len(c_ids))
        p_ids = p_ids[len(p_ids) - keep_p:]
        c_ids = c_ids[: args.max_len]
    ids = p_ids + c_ids
    labels = [-100] * len(p_ids) + list(c_ids)
    return {"input_ids": ids, "labels": labels, "length": len(ids)}

ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=8)
print("num examples:", len(ds))
lens = ds["length"]
print("len stats: max", max(lens), "mean", sum(lens)/len(lens))

# sanity print of first example decoded
ex0 = ds[0]
print("=== decoded input[0] ===")
print(tok.decode(ex0["input_ids"]))
sup = [t for t,l in zip(ex0["input_ids"], ex0["labels"]) if l != -100]
print("=== supervised span[0] ===")
print(tok.decode(sup))

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
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

try:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    print("loaded as Gemma3ForConditionalGeneration")
except Exception as e:
    print("fallback CausalLM:", e)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager")

model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

targs = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bs,
    gradient_accumulation_steps=args.accum,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=args.warmup,
    logging_steps=20,
    bf16=True,
    optim="adamw_8bit",
    save_strategy=("steps" if args.save_steps>0 else "no"),
    save_steps=(args.save_steps if args.save_steps>0 else 500),
    report_to="none",
    gradient_checkpointing=True,
    group_by_length=True,
    length_column_name="length",
    weight_decay=0.0,
    max_grad_norm=1.0,
    dataloader_num_workers=4,
)

trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(pad_id))
trainer.train()
trainer.save_model(args.output)
tok.save_pretrained(args.output)
print("SAVED to", args.output)
