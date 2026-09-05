#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt for GSM8K.

Everything the grader sees is reproduced here byte-for-byte:
  * the chat template is templates/gemma3.jinja (the file evaluate.py passes to
    vLLM), loaded from disk and hashed - never the tokenizer's own template;
  * the user turn is inspect_evals' MATH_PROMPT_TEMPLATE;
  * the target ends with "ANSWER: <number>" followed by <end_of_turn> (token
    106), which is the terminator in generation_config.eos_token_id.

Sequences are packed into fixed-length blocks; prompt tokens carry label -100.
"""
import argparse
import hashlib
import json
import os
import random
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)

TASK = "/home/ben/task"
SNAPSHOT = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
            "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
TEMPLATE = f"{TASK}/templates/gemma3.jinja"


class Packed(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        ids, lab = self.blocks[i]
        return {"input_ids": ids, "labels": lab,
                "attention_mask": [1] * len(ids)}


def collate(feats):
    return {k: torch.tensor([f[k] for f in feats], dtype=torch.long)
            for k in ("input_ids", "labels", "attention_mask")}


def fewshot_prefix(demos):
    return "\n\n".join(
        f"{d['q']}\n\nReasoning:\n{d['r']}\n\nANSWER: {d['a']}" for d in demos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{TASK}/data/pool_clean.jsonl")
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rows", type=int, default=10**9)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--block", type=int, default=3072)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--fewshot-frac", type=float, default=0.10)
    ap.add_argument("--fewshot-kmax", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--liger", action="store_true")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- template: the exact bytes the grader passes to vLLM ---------------
    tpl = open(TEMPLATE).read()
    tpl_sha = hashlib.sha256(tpl.encode()).hexdigest()
    print(f"chat template sha256={tpl_sha}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.parent)
    tok.chat_template = tpl
    EOT = "<end_of_turn>"
    assert tok.convert_tokens_to_ids(EOT) == 106

    # ---- few-shot demo bank, from the GSM8K TRAIN split only ---------------
    demos = []
    if args.fewshot_frac > 0:
        import datasets
        dev_q = {json.loads(l)["question"]
                 for l in open(f"{TASK}/data/dev_internal.jsonl")}
        for r in datasets.load_dataset("openai/gsm8k", "main", split="train"):
            if r["question"].strip() in dev_q:
                continue
            reasoning, _, a = r["answer"].rpartition("####")
            demos.append({"q": r["question"].strip(),
                          "r": reasoning.strip(), "a": a.strip()})
        rng.shuffle(demos)
        demos = demos[:3000]

    # ---- tokenise ----------------------------------------------------------
    rows = []
    with open(args.data) as fh:
        for line in fh:
            rows.append(json.loads(line))
    rng.shuffle(rows)
    rows = rows[: args.n_rows]

    examples, n_trunc, lens = [], 0, []
    for i, r in enumerate(rows):
        msgs = []
        if demos and rng.random() < args.fewshot_frac:
            k = rng.randint(1, args.fewshot_kmax)
            msgs.append({"role": "system",
                         "content": fewshot_prefix(rng.sample(demos, k))})
        msgs.append({"role": "user", "content": r["prompt"]})
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        tgt = r["target"].strip()
        if not tgt.endswith(EOT):
            tgt += EOT
        t_ids = tok(tgt, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > args.block:
            n_trunc += 1
            continue
        examples.append((p_ids, t_ids))
        lens.append(len(p_ids) + len(t_ids))
        if i == 0:
            print("---- rendered example ----")
            print(prompt + tgt)
            print("--------------------------", flush=True)

    lens.sort()
    print(f"examples={len(examples)} dropped_too_long={n_trunc} "
          f"len p50={lens[len(lens)//2]} p95={lens[int(.95*len(lens))]} "
          f"max={lens[-1]}", flush=True)
    assert n_trunc / max(1, len(rows)) < 0.02, "more than 2% of rows truncate"

    # ---- pack --------------------------------------------------------------
    blocks, ids, lab = [], [], []
    for p_ids, t_ids in examples:
        e_ids = p_ids + t_ids
        e_lab = [-100] * len(p_ids) + t_ids
        if len(ids) + len(e_ids) > args.block:
            pad = args.block - len(ids)
            blocks.append((ids + [tok.pad_token_id] * pad, lab + [-100] * pad))
            ids, lab = [], []
        ids += e_ids
        lab += e_lab
    if ids:
        pad = args.block - len(ids)
        blocks.append((ids + [tok.pad_token_id] * pad, lab + [-100] * pad))
    n_loss = sum(sum(1 for x in l if x != -100) for _, l in blocks)
    print(f"blocks={len(blocks)} tokens={len(blocks)*args.block} "
          f"loss_tokens={n_loss}", flush=True)
    assert n_loss > 0

    # ---- model -------------------------------------------------------------
    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3()
        print("liger applied", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn)
    for n, p in model.named_parameters():
        if n.startswith("model.vision_tower") or \
           n.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    model.config.use_cache = False
    # A parent that already carries the greedy decode config (temperature 0.0
    # with do_sample False) makes transformers refuse to re-save its
    # GenerationConfig, which kills the run at the first checkpoint. Clear the
    # sampling fields on the in-memory config; the file is rewritten below.
    for _f in ("temperature", "top_k", "top_p"):
        if getattr(model.generation_config, _f, None) is not None:
            setattr(model.generation_config, _f, None)
    model.generation_config.do_sample = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.micro_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 1000000,
        save_total_limit=None,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, data_collator=collate,
                      train_dataset=Packed(blocks))
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.parent, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, f))
    # greedy decode: vLLM reads generation_config.json for server defaults and
    # evaluate.py never sends a temperature.
    json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
               "cache_implementation": "hybrid",
               "do_sample": False, "temperature": 0.0},
              open(os.path.join(final, "generation_config.json"), "w"), indent=2)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
