#!/usr/bin/env python3
import argparse, json, os, math, random
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig,
                          Trainer, TrainingArguments, set_seed)

BASE = os.environ.get("PTB_BASE_MODEL_SNAPSHOT",
                      "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")
TEMPLATE = "templates/gemma3.jinja"


def build_tokenizer(path=BASE):
    tok = AutoTokenizer.from_pretrained(path)
    tok.chat_template = open(TEMPLATE).read()
    return tok


class Collator:
    def __init__(self, pad_id, pad_to=16):
        self.pad_id = pad_id
        self.pad_to = pad_to

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        maxlen = int(math.ceil(maxlen / self.pad_to) * self.pad_to)
        ids, labels, mask = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            p = maxlen - n
            ids.append(f["input_ids"] + [self.pad_id] * p)
            labels.append(f["labels"] + [-100] * p)
            mask.append([1] * n + [0] * p)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/sft.jsonl")
    ap.add_argument("--out", default="work/run1")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    tok = build_tokenizer()
    rows = [json.loads(l) for l in open(args.data)]
    if args.max_samples > 0:
        random.Random(0).shuffle(rows)
        rows = rows[:args.max_samples]
    print("rows", len(rows), flush=True)

    def encode(batch):
        out_ids, out_lab, out_len = [], [], []
        for msgs in batch["messages"]:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            full = prompt + msgs[-1]["content"] + "<end_of_turn>\n"
            pid = tok(prompt, add_special_tokens=False)["input_ids"]
            fid = tok(full, add_special_tokens=False)["input_ids"]
            if len(fid) > args.max_len or len(fid) <= len(pid):
                out_ids.append([]); out_lab.append([]); out_len.append(0); continue
            lab = [-100] * len(pid) + fid[len(pid):]
            out_ids.append(fid); out_lab.append(lab); out_len.append(len(fid))
        return {"input_ids": out_ids, "labels": out_lab, "length": out_len}

    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    ds = ds.map(encode, batched=True, batch_size=200, num_proc=16, remove_columns=["messages"])
    ds = ds.filter(lambda x: x["length"] > 0, num_proc=16)
    print("encoded", len(ds), "tokens", sum(ds["length"]), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=torch.float32, attn_implementation="sdpa")
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        for p in model.model.vision_tower.parameters():
            p.requires_grad = False
        for p in model.model.multi_modal_projector.parameters():
            p.requires_grad = False
    model.config.use_cache = False
    gc = model.generation_config
    gc.do_sample = True
    gc.temperature = 1.0
    gc.top_p = 0.95
    gc.top_k = 64
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable", ntr / 1e9, "B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_steps=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        adam_beta2=0.95,
        bf16=True,
        optim="adamw_8bit",
        use_liger_kernel=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        logging_steps=10,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=4,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    model.config.use_cache = True
    model = model.to(torch.bfloat16)
    trainer.save_model(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    print("saved", args.out + "/final")


if __name__ == "__main__":
    main()
