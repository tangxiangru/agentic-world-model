#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt for GSM8K.

Prompts are rendered with the *grader's own* chat template
(templates/gemma3.jinja, sha256 pinned below) and the grader's prompt
template, so training and evaluation see byte-identical strings.
Targets end with <end_of_turn> (token 106), which is in the model's
generation_config eos_token_id list, i.e. the token vLLM stops on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = "templates/gemma3.jinja"
TEMPLATE_SHA256 = "0e1b41ce7b8b2fc7e2c7a92f7bd8a8e1"  # filled at runtime, printed

# verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"


def load_template() -> str:
    with open(TEMPLATE_PATH) as f:
        t = f.read()
    print("chat template sha256:", hashlib.sha256(t.encode()).hexdigest())
    return t


def fewshot_block(examples) -> str:
    """Mirrors inspect_evals.gsm8k.sample_to_fewshot exactly."""
    return "\n\n".join(
        f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {ans}" for q, reasoning, ans in examples
    )


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, p_fewshot, fewshot_pool, seed=0, limit=None):
        self.rows = []
        rng = random.Random(seed)
        raw = [json.loads(l) for l in open(path)]
        if limit:
            raw = raw[:limit]
        n_trunc = 0
        lens = []
        for r in raw:
            msgs = []
            user = MATH_PROMPT_TEMPLATE.format(prompt=r["question"])
            if fewshot_pool and rng.random() < p_fewshot:
                k = rng.choice([2, 4, 6, 8, 10])
                shots = rng.sample(fewshot_pool, k)
                msgs.append({"role": "system", "content": fewshot_block(shots)})
            msgs.append({"role": "user", "content": user})
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            tgt = r["target"].strip()
            if not tgt.endswith(END_OF_TURN):
                tgt += END_OF_TURN
            t_ids = tok(tgt, add_special_tokens=False)["input_ids"]
            lens.append(len(p_ids) + len(t_ids))
            if len(p_ids) + len(t_ids) > max_len:
                n_trunc += 1
                continue
            self.rows.append((p_ids, t_ids))
        lens.sort()
        print(
            f"{path}: kept {len(self.rows)}/{len(raw)}  dropped(too long) {n_trunc} "
            f"({n_trunc/max(1,len(raw)):.3%})  p50={lens[len(lens)//2]} "
            f"p99={lens[int(len(lens)*0.99)]} max={lens[-1]}"
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, t = self.rows[i]
        ids = p + t
        labels = [-100] * len(p) + list(t)
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, mask = [], [], []
        for f in feats:
            d = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * d)
            labels.append(f["labels"] + [-100] * d)
            mask.append([1] * len(f["input_ids"]) + [0] * d)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--out", default="ckpts/exp-02")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--p-fewshot", type=float, default=0.2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--liger", action="store_true")
    ap.add_argument("--max-tokens-per-batch", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    template = load_template()
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = template
    tok.padding_side = "right"

    # few-shot exemplar pool: GSM8K train items that are NOT in the probe split
    from datasets import load_dataset

    g = load_dataset("openai/gsm8k", "main")["train"]
    split = json.load(open("data/split_idx.json"))
    pool = []
    for i in split["train_idx"][:2000]:
        r = g[i]
        body, ans = r["answer"].rsplit("####", 1)
        pool.append((r["question"], body.strip(), ans.strip()))

    ds = SFTData(
        args.data, tok, args.max_seq_len, args.p_fewshot, pool, seed=args.seed, limit=args.limit
    )

    if args.dry_run:
        p, t = ds.rows[0]
        print("=" * 30, "RENDERED EXAMPLE", "=" * 30)
        print(tok.decode(p))
        print("-" * 30, "TARGET", "-" * 30)
        print(tok.decode(t))
        print("last target token id:", t[-1], repr(tok.decode([t[-1]])))
        # a few-shot example too
        for p, t in ds.rows:
            if len(p) > 900:
                print("=" * 20, "FEWSHOT-PREFIXED PROMPT (head/tail)", "=" * 20)
                print(tok.decode(p)[:600], "\n ... \n", tok.decode(p)[-600:])
                break
        return

    from transformers import Gemma3ForConditionalGeneration

    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger fused-linear-CE patched into gemma3")

    cfg = AutoConfig.from_pretrained(args.parent)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn, config=cfg
    )
    model.config.use_cache = False
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen vision params: {n_frozen/1e6:.0f}M ; trainable: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to=[],
        seed=args.seed,
        group_by_length=True,
        length_column_name=None,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the processor bits vLLM wants for a multimodal gemma3 config
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            with open(src) as f:
                data = f.read()
            with open(os.path.join(final, fn), "w") as f:
                f.write(data)
    print("saved", final)


if __name__ == "__main__":
    main()
