#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on gsm8k-style CoT data.

The prompt/target strings are rendered with the SAME jinja template the grader
uses (templates/gemma3.jinja) and the SAME prompt wrapper the inspect task uses,
so training and grading render byte-identical strings (pitfall: template_unreachable).
Loss is computed on the completion only.
"""
import argparse, hashlib, json, os, random, sys

import torch
from torch.utils.data import Dataset
from transformers import (AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)

# verbatim from inspect_evals/gsm8k/gsm8k.py (MATH_PROMPT_TEMPLATE)
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

STOP = "<end_of_turn>"


def fewshot_block(q, reasoning, ans):
    return f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {ans}"


class SFTData(Dataset):
    def __init__(self, rows, tok, template, max_len, fewshot_pool, fewshot_p, seed=0):
        self.tok, self.template, self.max_len = tok, template, max_len
        self.rows, self.fewshot_pool, self.fewshot_p = rows, fewshot_pool, fewshot_p
        self.rng = random.Random(seed)
        self.cache = {}
        self.lengths = []
        for i in range(len(rows)):
            ex = self._build(i)
            self.cache[i] = ex
            self.lengths.append(len(ex["input_ids"]))

    def _build(self, i):
        r = self.rows[i]
        msgs = []
        if self.fewshot_pool and self.rng.random() < self.fewshot_p:
            k = self.rng.choice([1, 2, 3, 4])
            shots = self.rng.sample(self.fewshot_pool, k)
            msgs.append({"role": "system", "content": "\n\n".join(shots)})
        msgs.append({"role": "user",
                     "content": MATH_PROMPT_TEMPLATE.replace("{prompt}", r["question"])})
        prompt = self.tok.apply_chat_template(msgs, chat_template=self.template,
                                              tokenize=False, add_generation_prompt=True)
        target = r.get("completion") or (r["solution"].strip() + "\n\nANSWER: " + r["answer"] + STOP)
        p_ids = self.tok(prompt, add_special_tokens=False).input_ids
        t_ids = self.tok(target, add_special_tokens=False).input_ids
        ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        if len(ids) > self.max_len:      # keep the completion, drop the front of the prompt
            ids = ids[-self.max_len:]
            labels = labels[-self.max_len:]
        return {"input_ids": ids, "labels": labels}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if i >= len(self.rows) or i < 0:
            raise IndexError(i)
        return self.cache[i]


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Length-bucketed batches with a cap on padded tokens per micro-batch.

    Fixed batch sizes OOM on this model: the lm head is 262k wide and
    transformers upcasts logits to fp32, so memory scales with
    batch*seq*262144*4 bytes. Capping padded tokens per batch keeps that flat.
    """

    def __init__(self, lengths, budget, max_bs, seed):
        self.batches = []
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        cur, curmax = [], 0
        for i in order:
            m = max(curmax, lengths[i])
            if cur and ((len(cur) + 1) * m > budget or len(cur) + 1 > max_bs):
                self.batches.append(cur)
                cur, curmax = [i], lengths[i]
            else:
                cur, curmax = cur + [i], m
        if cur:
            self.batches.append(cur)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        r = random.Random(self.seed + self.epoch)
        self.epoch += 1
        b = list(self.batches)
        r.shuffle(b)
        return iter(b)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            p = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * p)
            labels.append(f["labels"] + [-100] * p)
            mask.append([1] * len(f["input_ids"]) + [0] * p)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


class BudgetTrainer(Trainer):
    batch_sampler = None

    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_sampler=self.batch_sampler,
            collate_fn=self.data_collator, num_workers=2, pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="templates/gemma3.jinja")
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=32)          # max rows per micro-batch
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--fewshot-p", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template = open(args.template).read()
    print("template sha256:", hashlib.sha256(template.encode()).hexdigest(), flush=True)

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit_rows:
        rows = rows[:args.limit_rows]
    print(f"{len(rows)} training rows", flush=True)

    # few-shot pool: real GSM8K TRAIN items, rendered exactly like the harness does
    fewshot_pool = []
    fp = "data/fewshot_pool.jsonl"
    if os.path.exists(fp) and args.fewshot_p > 0:
        for l in open(fp):
            d = json.loads(l)
            fewshot_pool.append(d["shot"])
    print(f"{len(fewshot_pool)} few-shot blocks", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTData(rows, tok, template, args.max_seq_len, fewshot_pool, args.fewshot_p, args.seed)
    ls = sorted(ds.lengths)
    print(f"token lengths: p50={ls[len(ls)//2]} p90={ls[int(.9*len(ls))]} "
          f"p99={ls[int(.99*len(ls))]} max={ls[-1]} "
          f"truncated={sum(1 for x in ds.lengths if x >= args.max_seq_len)}", flush=True)
    ex = ds[0]
    print("=== example prompt+target ===", flush=True)
    print(repr(tok.decode(ex["input_ids"])[-1200:]), flush=True)
    print("=== loss-carrying part ===", flush=True)
    print(repr(tok.decode([t for t in ex["labels"] if t != -100])), flush=True)
    assert tok.decode([t for t in ex["labels"] if t != -100]).endswith(STOP)
    if args.dry_run:
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model.config.use_cache = False
    for n, p in model.named_parameters():
        if n.startswith("model.vision_tower") or n.startswith("model.multi_modal_projector") \
           or n.startswith("vision_tower") or n.startswith("multi_modal_projector"):
            p.requires_grad_(False)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {ntr/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
    )
    bsampler = TokenBudgetBatches(ds.lengths, args.token_budget, args.bs, args.seed)
    print(f"{len(bsampler)} micro-batches/epoch -> "
          f"{len(bsampler)//args.grad_accum} optimiser steps/epoch", flush=True)
    trainer = BudgetTrainer(model=model, args=targs, train_dataset=ds,
                            data_collator=Collator(tok.pad_token_id))
    trainer.batch_sampler = bsampler
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    try:
        proc = AutoProcessor.from_pretrained(args.model)
        proc.save_pretrained(final)
    except Exception as e:
        print("processor save failed:", e, flush=True)
    tok.save_pretrained(final)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
