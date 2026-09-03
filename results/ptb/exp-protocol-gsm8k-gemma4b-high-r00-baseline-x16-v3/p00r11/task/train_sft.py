#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt for GSM8K.

Rendering uses templates/gemma3.jinja -- the exact file the grader passes to
vLLM -- so training strings are byte-identical to grading strings, and every
target ends with <end_of_turn> (the terminator in the model's
generation_config eos_token_id list).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

TEMPLATE_PATH = "templates/gemma3.jinja"


class SFTRows(Dataset):
    def __init__(self, path, tok, template, max_seq_len, report=True):
        self.rows = []
        self.tok = tok
        n_trunc = 0
        lens = []
        with open(path) as f:
            raw = [json.loads(l) for l in f]
        for r in raw:
            msgs = []
            if r.get("system"):
                msgs.append({"role": "system", "content": r["system"]})
            msgs.append({"role": "user", "content": r["prompt"]})
            prompt_str = tok.apply_chat_template(
                msgs, chat_template=template, tokenize=False,
                add_generation_prompt=True,
            )
            full_str = prompt_str + r["completion"] + "\n"  # completion ends with <end_of_turn>
            p_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
            f_ids = tok(full_str, add_special_tokens=False)["input_ids"]
            lens.append(len(f_ids))
            if len(f_ids) > max_seq_len:
                n_trunc += 1
                continue
            labels = list(f_ids)
            labels[: len(p_ids)] = [-100] * len(p_ids)
            self.rows.append({"input_ids": f_ids, "labels": labels})
        if report:
            lens.sort()
            print(
                f"[data] {path}: kept {len(self.rows)}/{len(raw)} rows, "
                f"dropped {n_trunc} over max_seq_len={max_seq_len} "
                f"({n_trunc/max(1,len(raw)):.3%}); "
                f"len p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}",
                flush=True,
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


class ChunkedCETrainer(Trainer):
    """Cross-entropy over supervised tokens only, in checkpointed chunks.

    Gemma3's own loss path materialises [B, T, 262208] logits and an fp32 copy
    of them; at B=16, T=1536 that is a 17 GiB allocation and OOMs an 80 GB H100.
    Here the LM head is applied only to the completion tokens that carry loss,
    in chunks whose logits are recomputed in the backward pass.
    """

    CHUNK = 2048

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        base = model.model if hasattr(model, "model") else model.module.model
        head = model.lm_head if hasattr(model, "lm_head") else model.module.lm_head
        hidden = base(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state
        h = hidden[:, :-1, :]
        t = labels[:, 1:]
        keep = t != -100
        h = h[keep]
        t = t[keep]
        n = t.numel()

        def chunk_loss(hc, tc):
            return F.cross_entropy(head(hc).float(), tc, reduction="sum")

        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, n, self.CHUNK):
            hc, tc = h[i : i + self.CHUNK], t[i : i + self.CHUNK]
            total = total + (
                checkpoint(chunk_loss, hc, tc, use_reentrant=False)
                if self.model.training
                else chunk_loss(hc, tc)
            )
        denom = num_items_in_batch if num_items_in_batch is not None else n
        loss = total / denom
        return (loss, None) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    template = open(TEMPLATE_PATH).read()
    print("[template] sha256", hashlib.sha256(template.encode()).hexdigest()[:16], flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = SFTRows(args.data, tok, template, args.max_seq_len)
    ex = ds[0]
    print("[example]", json.dumps(tok.decode(ex["input_ids"]))[:1200], flush=True)
    sup = [i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    print("[supervised tail]", json.dumps(tok.decode(sup[-40:])), flush=True)
    assert ex["input_ids"][-2:] == tok("<end_of_turn>\n", add_special_tokens=False)["input_ids"], \
        "targets must end with the grading stop token"
    if args.dry_run:
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    # text-only task: keep the vision stack frozen so the checkpoint stays
    # loadable by the grader's vLLM but costs no optimizer state
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params {n_train/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        group_by_length=True,
        length_column_name=None,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        save_safetensors=True,
    )
    trainer = ChunkedCETrainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the processor/vision preprocessing files so vLLM loads the
    # multimodal config exactly as it loads the base snapshot
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, fn)
        if os.path.exists(src):
            with open(src) as a, open(os.path.join(final, fn), "w") as b:
                b.write(a.read())
    print("[done] saved", final, flush=True)


if __name__ == "__main__":
    main()
