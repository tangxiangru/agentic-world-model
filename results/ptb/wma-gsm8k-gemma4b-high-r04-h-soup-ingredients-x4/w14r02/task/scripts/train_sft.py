#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on GSM8K-style data.

Renders every row through the *grader's* chat template (templates/gemma3.jinja)
so training and grading agree byte-for-byte, masks the prompt, and trains on the
completion plus the terminator the grader stops on (<end_of_turn>).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--no-liger", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="tokenize + report, no training")
    return p.parse_args()


def build_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    template = open(TEMPLATE_PATH).read()
    print("template sha256:", hashlib.sha256(template.encode()).hexdigest())
    tok.chat_template = template
    return tok


def render(tok, prompt: str, completion: str) -> tuple[str, str]:
    """Return (prompt_text, full_text) exactly as the grader's template renders them."""
    ptxt = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ftxt = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return ptxt, ftxt


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def main():
    args = parse_args()
    tok = build_tokenizer(args.model)

    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    print("eos_token:", tok.eos_token, tok.eos_token_id, " <end_of_turn>:", eot)

    rows = [json.loads(l) for l in open(args.data)]
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"{len(rows)} raw rows")

    # one full render, printed, so a human can see what the model actually sees
    p0, f0 = render(tok, rows[0]["prompt"], rows[0]["completion"])
    print("=" * 30, "RENDERED PROMPT", "=" * 30)
    print(repr(p0))
    print("=" * 30, "RENDERED FULL", "=" * 30)
    print(repr(f0[len(p0) :]))
    assert f0.startswith(p0), "prompt is not a prefix of the full rendering"
    assert f0.rstrip("\n").endswith("<end_of_turn>"), "target does not end with the grader's stop token"

    feats, too_long, lens = [], 0, []
    for r in rows:
        ptxt, ftxt = render(tok, r["prompt"], r["completion"])
        pid = tok(ptxt, add_special_tokens=False)["input_ids"]
        fid = tok(ftxt.rstrip("\n"), add_special_tokens=False)["input_ids"]
        if fid[-1] != eot:
            raise RuntimeError("row does not end on <end_of_turn>")
        if len(fid) > args.max_seq_len:
            too_long += 1
            continue
        lens.append(len(fid))
        lab = [-100] * len(pid) + fid[len(pid) :]
        feats.append({"input_ids": fid, "labels": lab, "length": len(fid)})

    lens.sort()
    print(
        f"kept {len(feats)}  dropped-too-long {too_long} "
        f"({too_long / max(1, len(rows)):.3%})  "
        f"p50={lens[len(lens) // 2]} p99={lens[int(len(lens) * 0.99)]} max={lens[-1]}"
    )
    if too_long / max(1, len(rows)) > 0.02:
        raise SystemExit("more than 2% of rows exceed max_seq_len - raise it (pitfall seq_len_truncation)")

    n_loss = sum(sum(1 for x in f["labels"] if x != -100) for f in feats[:2000])
    print(f"avg loss tokens per row (first 2000): {n_loss / min(2000, len(feats)):.1f}")

    if args.dry_run:
        return

    if not args.no_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger applied")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    )
    print(type(model).__name__)
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen vision params: {n_frozen / 1e6:.1f}M")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=feats,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok_out = AutoTokenizer.from_pretrained(args.model)
    tok_out.save_pretrained(final)
    # greedy decoding: vLLM reads generation_config.json for default sampling params
    gc = json.load(open(os.path.join(args.model, "generation_config.json")))
    # vLLM's ModelConfig.get_diff_sampling_param reads ONLY
    # {repetition_penalty, temperature, top_k, top_p, min_p, max_new_tokens}
    # from generation_config.json -- do_sample is ignored. Greedy therefore has
    # to be spelled out as temperature 0.0 / top_p 1.0 / top_k -1.
    # top_k is dropped rather than set to a -1 sentinel: a -1 in a checkpoint's
    # generation_config can make a later save_pretrained raise, and temperature
    # 0.0 already forces greedy in vLLM.
    gc.pop("top_k", None)
    gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0})
    gc["eos_token_id"] = [1, 106]  # <eos> and <end_of_turn>; Trainer can collapse this to 1
    json.dump(gc, open(os.path.join(final, "generation_config.json"), "w"), indent=2)
    # the grader needs the processor for a Gemma3ForConditionalGeneration checkpoint
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, f)
        if os.path.exists(src):
            import shutil

            shutil.copy(src, os.path.join(final, f))
    print("saved", final)


if __name__ == "__main__":
    main()
