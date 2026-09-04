#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt on GSM8K-style data.

Trains only the language model (vision tower + projector frozen). Loss is
computed on the assistant response tokens only. Output format matches the
eval: reasoning + final line 'ANSWER: <number>'.
"""
import os, json, argparse, random
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForImageTextToText,
                          Trainer, TrainingArguments)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train_data.jsonl")
    p.add_argument("--out", default="sft_out")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--maxlen", type=int, default=1024)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--wrap_prob", type=float, default=1.0,
                   help="prob of wrapping question in MATH_PROMPT_TEMPLATE")
    return p.parse_args()

class SFTDataset(Dataset):
    def __init__(self, path, tok, maxlen, limit, wrap_prob):
        self.tok = tok
        self.maxlen = maxlen
        self.wrap_prob = wrap_prob
        self.rows = []
        with open(path) as f:
            for line in f:
                self.rows.append(json.loads(line))
        if limit > 0:
            self.rows = self.rows[:limit]
        self.bos = tok.bos_token  # <bos>
        self.eot = "<end_of_turn>"
        random.seed(1234)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        resp = r["response"].strip()
        if "user" in r:
            user = r["user"]
        else:
            q = r["question"].strip()
            if random.random() < self.wrap_prob:
                user = MATH_PROMPT_TEMPLATE.format(prompt=q)
            else:
                user = q
        # Gemma chat format (matches gemma3.jinja, no system message here)
        head = f"{self.bos}<start_of_turn>user\n"
        tail = "<end_of_turn>\n<start_of_turn>model\n"
        post = f"{resp}<end_of_turn>\n"
        head_ids = self.tok(head, add_special_tokens=False)["input_ids"]
        user_ids = self.tok(user, add_special_tokens=False)["input_ids"]
        tail_ids = self.tok(tail, add_special_tokens=False)["input_ids"]
        post_ids = self.tok(post, add_special_tokens=False)["input_ids"]
        # Always keep full response (post_ids). Truncate the user prefix from the
        # left if needed so the '<end_of_turn>' stop token is never lost.
        fixed = len(head_ids) + len(tail_ids) + len(post_ids)
        budget = self.maxlen - fixed
        if budget < 0:
            # response alone too long: hard truncate from right (rare)
            post_ids = post_ids[: self.maxlen - len(head_ids) - len(tail_ids)]
            user_ids = []
        elif len(user_ids) > budget:
            user_ids = user_ids[-budget:]
        prompt_ids = head_ids + user_ids + tail_ids
        full_ids = prompt_ids + post_ids
        labels = [-100] * len(prompt_ids) + list(post_ids)
        return {"input_ids": full_ids, "labels": labels}

class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]; lab = b["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id]*pad)
            labels.append(lab + [-100]*pad)
            attn.append([1]*len(ids) + [0]*pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        SNAP, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    # Freeze vision tower + multimodal projector; train language model only.
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        if ("vision_tower" in name) or ("multi_modal_projector" in name):
            p.requires_grad_(False); n_freeze += p.numel()
        else:
            p.requires_grad_(True); n_train += p.numel()
    print(f"trainable params: {n_train/1e9:.2f}B  frozen: {n_freeze/1e9:.2f}B")

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ds = SFTDataset(args.data, tok, args.maxlen, args.limit, args.wrap_prob)
    print("dataset size:", len(ds))
    collator = Collator(tok.pad_token_id)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to=[],
        dataloader_num_workers=4,
        gradient_checkpointing=False,  # enabled manually above
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("saved to", args.out)

if __name__ == "__main__":
    main()
