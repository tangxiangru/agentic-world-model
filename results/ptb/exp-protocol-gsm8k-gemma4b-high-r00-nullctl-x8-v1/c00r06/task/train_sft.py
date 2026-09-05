#!/usr/bin/env python3
"""Full-parameter SFT of gemma-3-4b-pt for GSM8K-style math reasoning."""
import argparse, json, math, os, random, sys

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def load_fewshot_pool():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for r in ds:
        parts = r["answer"].split("####")
        target = parts[-1].strip()
        reasoning = "####".join(parts[:-1]).strip()
        pool.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return pool


class SFTData(Dataset):
    def __init__(self, path, tok, chat_template, max_len=2048,
                 fewshot_frac=0.15, fewshot_pool=None, seed=0, limit=None):
        self.ex = []
        rows = [json.loads(l) for l in open(path)]
        if limit:
            rows = rows[:limit]
        rng = random.Random(seed)
        self.tok = tok
        n_skip = 0
        for i, r in enumerate(rows):
            use_fs = fewshot_pool is not None and rng.random() < fewshot_frac
            msgs = []
            if use_fs:
                k = rng.choice([10, 10, 10, 5, 2])
                shots = rng.sample(fewshot_pool, k)
                msgs.append({"role": "system", "content": "\n\n".join(shots)})
            msgs.append({"role": "user",
                         "content": PROMPT_TEMPLATE.format(prompt=r["question"])})
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True,
                                             chat_template=chat_template)
            resp = r["response"].strip() + "<end_of_turn>"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            r_ids = tok(resp, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(r_ids) > max_len:
                n_skip += 1
                continue
            self.ex.append((p_ids, r_ids))
        print(f"dataset: {len(self.ex)} examples, skipped {n_skip} too-long", flush=True)
        self.lengths = [len(a) + len(b) for a, b in self.ex]
        print("total tokens:", sum(self.lengths), flush=True)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        p, r = self.ex[i]
        ids = p + r
        labels = [-100] * len(p) + list(r)
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = ((n + 7) // 8) * 8
        input_ids, labels, attn = [], [], []
        for f in feats:
            d = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * d)
            labels.append(f["labels"] + [-100] * d)
            attn.append([1] * len(f["input_ids"]) + [0] * d)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--out", default="ckpt/sft1")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--fewshot-frac", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--fp32-master", type=int, default=1)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--liger", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    chat_template = open("templates/gemma3.jinja").read()

    pool = load_fewshot_pool() if args.fewshot_frac > 0 else None
    ds = SFTData(args.data, tok, chat_template, args.max_len,
                 args.fewshot_frac, pool, args.seed, args.limit)

    dtype = torch.float32 if args.fp32_master else torch.bfloat16
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=dtype, attn_implementation=args.attn)
    model.config.use_cache = False
    # freeze vision tower / projector (text-only training)
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print("frozen params:", n_frozen / 1e6, "M", flush=True)
    print("trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9, "B", flush=True)

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
        adam_beta2=0.95,
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        use_liger_kernel=bool(args.liger),
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
    )
    # provide lengths for the length grouped sampler
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    model = model.to(torch.bfloat16)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    # copy processor/config extras so vLLM can load it like the base model
    import shutil
    for f in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "cache_implementation": "hybrid",
                   "do_sample": False, "temperature": 0.0,
                   "transformers_version": "4.50.0.dev0"}, f, indent=2)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
