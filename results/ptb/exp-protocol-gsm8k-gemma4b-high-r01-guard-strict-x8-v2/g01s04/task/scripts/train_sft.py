#!/usr/bin/env python3
"""Completion-only SFT of gemma-3-4b-pt, rendered with the grader's template.

Prompt/target are produced by tokenizer.apply_chat_template using
templates/gemma3.jinja (the exact file evaluate.py hands to vLLM), so the
strings the model trains on are byte-identical to the strings it is graded on.
Loss is on the completion only; every target is terminated with <end_of_turn>
(id 106), which is in the checkpoint's generation_config eos_token_id list.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    END_OF_TURN_ID,
    SNAPSHOT,
    build_messages,
    load_tokenizer,
    template_sha,
)


def encode(tok, row, max_len: int):
    msgs = build_messages(row["question"], row.get("fewshots") or None)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    # the target already carries the literal <end_of_turn> terminator
    t_ids = tok(row["target"].strip(), add_special_tokens=False)["input_ids"]
    assert t_ids[-1] == END_OF_TURN_ID, "target does not end in the stop token"
    ids = p_ids + t_ids
    if len(ids) > max_len:
        return None, len(ids)
    labels = [-100] * len(p_ids) + t_ids
    return {"input_ids": ids, "labels": labels}, len(ids)


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, mask = [], [], []
        for f in feats:
            pad = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            mask.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("chat template sha:", template_sha(), flush=True)
    tok = load_tokenizer()

    rows = []
    with open(args.data) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.n > 0:
        rows = rows[: args.n]

    feats, lens, dropped = [], [], 0
    for r in rows:
        enc, ln = encode(tok, r, args.max_seq_len)
        lens.append(ln)
        if enc is None:
            dropped += 1
            continue
        feats.append(enc)
    lens.sort()
    print(
        f"rows={len(rows)} kept={len(feats)} dropped={dropped} "
        f"({dropped / max(1, len(rows)):.3%}) "
        f"len p50={lens[len(lens) // 2]} p99={lens[int(len(lens) * 0.99)]} max={lens[-1]}",
        flush=True,
    )
    n_tokens = sum(len(f["input_ids"]) for f in feats)
    n_loss = sum(sum(1 for x in f["labels"] if x != -100) for f in feats)
    print(f"tokens={n_tokens / 1e6:.1f}M loss_tokens={n_loss / 1e6:.1f}M", flush=True)
    assert n_loss > 0

    # sanity: every kept row must end in the stop token and carry loss
    for f in feats[:200]:
        assert f["input_ids"][-1] == END_OF_TURN_ID
        assert f["labels"][-1] == END_OF_TURN_ID

    if args.dry_run:
        ex = feats[0]
        print("---- rendered example ----")
        print(tok.decode(ex["input_ids"]))
        print("---- loss span ----")
        print(tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100]))
        return

    from transformers import (
        Gemma3ForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # the vision stack is dead weight for a text task; freeze it so its
    # optimizer state does not sit in HBM
    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen={frozen / 1e9:.2f}B trainable={trainable / 1e9:.2f}B", flush=True)

    eff = args.bs * args.grad_accum
    steps_per_epoch = math.ceil(len(feats) / eff)
    print(f"effective batch={eff} steps/epoch={steps_per_epoch}", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=5,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        max_grad_norm=1.0,
        seed=args.seed,
        # gemma3's 262k vocab makes a materialised fp32 logit tensor
        # (bs x seq x 262144 x 4B) larger than HBM; liger's fused linear
        # cross-entropy never builds it
        use_liger_kernel=not args.no_liger,
        dataloader_num_workers=4,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        report_to=[],
    )

    for f in feats:
        f["length"] = len(f["input_ids"])

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=feats,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # Gemma3ForConditionalGeneration is a multimodal architecture: vLLM builds
    # a multimodal budget at startup and dies with "Can't load image processor"
    # unless the processor configs sit next to the weights
    # (pitfall: final_model_not_loadable)
    import shutil as _sh
    for _f in ("preprocessor_config.json", "processor_config.json"):
        _src = os.path.join(SNAPSHOT, _f)
        if os.path.exists(_src):
            _sh.copyfile(_src, os.path.join(final, _f))
    # keep the grader's own template out of the saved tokenizer_config so
    # evaluate.py's explicit --chat-template is unambiguous, but ship a
    # working one for anything that loads the model directly
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
