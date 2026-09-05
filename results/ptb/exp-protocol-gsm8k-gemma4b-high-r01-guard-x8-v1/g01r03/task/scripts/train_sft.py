"""Completion-only SFT for gemma-3-4b-pt on GSM8K-style data.

Prompts are rendered with the harness's own templates/gemma3.jinja (see
scripts/render.py) so the string the trainer sees is byte-for-byte the string
vLLM builds at grading time. Loss is on completion tokens only; every target
ends with <end_of_turn>, the terminator vLLM stops on (generation_config
eos_token_id = [1, 106]).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

PAD_ID = 0
IGNORE = -100


def build_dataset(path: str, tok, max_len: int, limit: int | None):
    prompts, completions = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            prompts.append(r["prompt"])
            completions.append(r["completion"])

    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    c_ids = tok(completions, add_special_tokens=False)["input_ids"]

    eot = tok.convert_tokens_to_ids(render.END_OF_TURN)
    input_ids, labels, lengths, dropped = [], [], [], 0
    for p, c in zip(p_ids, c_ids):
        assert c[-1] == eot, "target does not end with the stop token"
        if len(p) + len(c) > max_len:
            dropped += 1
            continue
        input_ids.append(p + c)
        labels.append([IGNORE] * len(p) + c)
        lengths.append(len(p) + len(c))
    print(f"rows kept {len(input_ids)}, dropped for length {dropped}, "
          f"tokens {sum(lengths)/1e6:.1f}M", flush=True)
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels, "length": lengths})


def collate(features):
    n = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for f in features:
        pad = n - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [PAD_ID] * pad)
        batch["labels"].append(f["labels"] + [IGNORE] * pad)
        batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=1280)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--param-dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--greedy-gen-config", type=int, default=1)
    args = ap.parse_args()

    assert render.template_hash() == render.TEMPLATE_SHA256, "gemma3.jinja changed"

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = build_dataset(args.data, tok, args.max_seq_len, args.limit)

    dtype = torch.float32 if args.param_dtype == "fp32" else torch.bfloat16
    cfg = AutoConfig.from_pretrained(args.model)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, attn_implementation=args.attn, config=cfg
    )
    model.config.use_cache = False

    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad = False
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {frozen/1e6:.0f}M, trainable {trainable/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
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
        save_only_model=True,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        use_liger_kernel=not args.no_liger,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()

    # save in bf16: config.json torch_dtype is bfloat16 and the grader loads with
    # vLLM, which would otherwise carry fp32 shards around for nothing
    final = os.path.join(args.out, "final")
    trainer.model.to(torch.bfloat16)
    trainer.save_model(final)
    tok.save_pretrained(final)
    # vLLM loads Gemma3ForConditionalGeneration as a multimodal model and needs the
    # image-processor files; save_pretrained/tokenizer do not write them, and without
    # them the grader's server dies with "Can't load image processor" (pitfall
    # final_model_not_loadable).
    import shutil
    for extra in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, extra)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(final, extra))
    if args.greedy_gen_config:
        # exp-03: vLLM reads this file as its server-side default sampling params and
        # inspect never sends a temperature, so greedy has to be written here.
        gc_path = os.path.join(final, "generation_config.json")
        gc = json.load(open(gc_path))
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc.pop("top_k", None)
        gc.pop("top_p", None)
        json.dump(gc, open(gc_path, "w"), indent=2)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
