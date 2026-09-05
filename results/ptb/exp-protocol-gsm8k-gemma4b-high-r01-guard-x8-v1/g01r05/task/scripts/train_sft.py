#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style CoT data.

Renders every training row with the *grader's own* chat template
(templates/gemma3.jinja, hash-checked) so the training string and the eval
string are byte-identical up to the question text. Loss is on the completion
only. Rows longer than --max-seq-len are dropped, never truncated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
TEMPLATE_SHA256 = None  # filled at runtime, printed for the record


def body_of(r: dict) -> str:
    """The model turn without the stop token (the data file stores it with)."""
    if "completion_body" in r:
        return r["completion_body"]
    return r["completion"]


def build_strings(tok, prompt: str, completion: str) -> tuple[str, str]:
    """Return (prefix_text, full_text) exactly as templates/gemma3.jinja renders."""
    prefix = (
        f"{tok.bos_token}<start_of_turn>user\n{prompt.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    full = f"{prefix}{completion.strip()}<end_of_turn>\n"
    return prefix, full


def verify_template(tok, prompt: str, completion: str) -> None:
    """Fail loudly if our hand-built string differs from the grader's template."""
    with open(TEMPLATE_PATH) as fh:
        template = fh.read()
    sha = hashlib.sha256(template.encode()).hexdigest()
    print(f"[template] {TEMPLATE_PATH} sha256={sha}")

    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
    rendered = tok.apply_chat_template(msgs, chat_template=template, tokenize=False)
    _, full = build_strings(tok, prompt, completion)
    if rendered != full:
        print("MISMATCH between grader template and training string")
        print("template :", repr(rendered[:400]), "...", repr(rendered[-200:]))
        print("training :", repr(full[:400]), "...", repr(full[-200:]))
        raise SystemExit(2)

    gen_msgs = [{"role": "user", "content": prompt}]
    gen_prompt = tok.apply_chat_template(
        gen_msgs, chat_template=template, tokenize=False, add_generation_prompt=True
    )
    prefix, _ = build_strings(tok, prompt, completion)
    if gen_prompt != prefix:
        print("MISMATCH between grader generation prompt and training prefix")
        print("template :", repr(gen_prompt[-300:]))
        print("training :", repr(prefix[-300:]))
        raise SystemExit(2)
    print("[template] training string == grader-rendered string (prefix and full)")


class SFTData(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        ids = b["input_ids"]
        lab = b["labels"]
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", type=str, default="adamw_bnb_8bit")
    ap.add_argument("--model-dtype", type=str, default="float32")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    rows_raw = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows_raw = rows_raw[: args.limit]
    verify_template(tok, rows_raw[0]["prompt"], body_of(rows_raw[0]))

    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    print(f"[tok] <end_of_turn> id={eot_id} eos={tok.eos_token}({tok.eos_token_id}) pad={tok.pad_token_id}")

    rows, dropped = [], 0
    n_marker_bad = 0
    for r in rows_raw:
        body = body_of(r)
        assert r["completion"] == body + "<end_of_turn>", "data file target must end with the stop token"
        prefix, full = build_strings(tok, r["prompt"], body)
        pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        if full_ids[: len(pre_ids)] != pre_ids:
            dropped += 1
            continue
        if len(full_ids) > args.max_seq_len:
            dropped += 1
            continue
        if body.count("ANSWER: ") != 1:
            n_marker_bad += 1
            continue
        labels = [-100] * len(pre_ids) + full_ids[len(pre_ids):]
        rows.append({"input_ids": full_ids, "labels": labels, "length": len(full_ids)})
    print(f"[data] kept {len(rows)} dropped {dropped} (len>{args.max_seq_len} or retok drift), bad-marker {n_marker_bad}")
    lens = sorted(r["length"] for r in rows)
    print(f"[data] tokens p50={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")
    # every target must end with the stop token
    assert all(r["input_ids"][-1] == eot_id or r["input_ids"][-2] == eot_id for r in rows[:2000])
    print("[data] all sampled targets end with <end_of_turn>")

    if not args.no_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True, cross_entropy=False)
        print("[liger] applied to gemma3 (fused linear cross entropy)")

    dtype = getattr(torch, args.model_dtype)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa"
    )
    # text-only task: freeze the vision tower and the multimodal projector
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") \
           or name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
            p.requires_grad = False
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {trainable/1e9:.3f}B, frozen(vision) {n_frozen/1e6:.1f}M, dtype={dtype}")
    model.config.use_cache = False
    # A parent checkpoint saved by this script carries the grader's greedy
    # generation_config (temperature 0.0 + do_sample False). HF's validator
    # rejects that combination on save, which kills every mid-run checkpoint.
    # Neutralise it here; the greedy config is rewritten after training.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.do_sample = False

    ds = SFTData(rows)
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim=args.optim,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()
    print("[mem] peak GiB", torch.cuda.max_memory_allocated()/2**30)

    if args.no_save:
        print("[save] skipped"); return
    final = os.path.join(args.out, "final")
    os.makedirs(final, exist_ok=True)
    model = trainer.model
    model.config.use_cache = True
    model.to(torch.bfloat16)
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    try:
        proc = AutoProcessor.from_pretrained(args.model)
        proc.save_pretrained(final)
    except Exception as e:  # pragma: no cover
        print("processor save failed:", e)
    # greedy decoding for the grader (vLLM reads temperature from generation_config.json)
    gen = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": False,
        "temperature": 0.0,
    }
    with open(os.path.join(final, "generation_config.json"), "w") as fh:
        json.dump(gen, fh, indent=2)
    print("[save] wrote", final)


if __name__ == "__main__":
    main()
