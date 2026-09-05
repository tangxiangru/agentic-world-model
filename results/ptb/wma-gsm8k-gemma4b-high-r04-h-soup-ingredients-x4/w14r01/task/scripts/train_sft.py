#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt on pre-rendered prompt/completion text.

The jsonl rows carry the *text of the user turn* (`prompt`) and the *text of the
model turn* (`completion`). This script renders them with the same string
templates the grader's templates/gemma3.jinja produces (verified byte-for-byte
by scripts/verify_format.py) and trains with loss on the completion only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import render_prompt, render_target  # noqa: E402

from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                p = tok(render_prompt(r["prompt"]), add_special_tokens=False)["input_ids"]
                c = tok(render_target(r["completion"]), add_special_tokens=False)["input_ids"]
                ids = p + c
                if len(ids) > max_seq_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p) + list(c)
                self.rows.append((ids, labels))
        print(f"[data] {len(self.rows)} rows kept, {n_trunc} dropped for length", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class PadCollator:
    def __init__(self, pad_id, pad_to_multiple_of=8):
        self.pad_id = pad_id
        self.m = pad_to_multiple_of

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = int(math.ceil(n / self.m) * self.m)
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
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
    ap.add_argument("--max-seq-len", type=int, default=2304)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-strategy", default="epoch")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--no-liger", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.no_liger:
        # fused linear cross-entropy: gemma-3's 262k vocab makes the logits
        # tensor (bs x seq x 262144, upcast to fp32) the peak allocation, and
        # it OOMs an 80 GB H100 at bs=16.  Liger never materialises it.
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("[model] liger gemma3 kernels applied", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    is_mm = cfg.architectures and "ConditionalGeneration" in cfg.architectures[0]

    if is_mm:
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=args.attn
        )
        for p in model.model.vision_tower.parameters():
            p.requires_grad = False
        for p in model.model.multi_modal_projector.parameters():
            p.requires_grad = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=args.attn
        )
    model.config.use_cache = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_train/1e9:.2f}B  (mm={is_mm})", flush=True)

    ds = SFTRows(args.data, tok, args.max_seq_len, args.limit)
    collator = PadCollator(tok.pad_token_id or 0)

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
        logging_steps=20,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=12,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tok,
    )

    # --- label-masking assertion on the first real collated batch ----------
    dl = trainer.get_train_dataloader()
    b = next(iter(dl))
    ids, lab = b["input_ids"], b["labels"]
    for r in range(min(4, ids.shape[0])):
        row_ids, row_lab = ids[r], lab[r]
        sup = (row_lab != -100).nonzero().flatten()
        assert len(sup) > 0, "row with zero supervised tokens"
        first, last = sup[0].item(), sup[-1].item()
        # the supervised span must be contiguous and end on <end_of_turn>
        assert last - first + 1 == len(sup), "supervised span is not contiguous"
        assert row_lab[last].item() == 106, (
            f"last supervised label is {row_lab[last].item()}, not 106 <end_of_turn>"
        )
        assert (row_lab[:first] == -100).all(), "prompt span is not fully masked"
        # the token just before the supervised span closes the generation prompt
        head = tok.decode(row_ids[max(0, first - 6) : first])
        assert head.endswith("<start_of_turn>model\n"), repr(head)
    print(
        f"[check] label masking ok on {min(4, ids.shape[0])} rows of the first batch; "
        f"batch shape {tuple(ids.shape)}, supervised tokens {(lab != -100).sum().item()}",
        flush=True,
    )

    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # Trainer has been seen to collapse gemma's eos_token_id [1, 106] to 1 when
    # it rewrites generation_config.json; 106 is <end_of_turn>, the only token
    # the grader's template stops on, so re-assert the base config verbatim.
    import shutil

    shutil.copy(
        os.path.join(args.model, "generation_config.json"),
        os.path.join(final, "generation_config.json"),
    )
    # keep the processor so vLLM can load the multimodal checkpoint unchanged
    if is_mm:
        try:
            from transformers import AutoProcessor

            AutoProcessor.from_pretrained(args.model).save_pretrained(final)
        except Exception as e:  # pragma: no cover
            print(f"[warn] processor not saved: {e}", flush=True)
    # every intermediate checkpoint must also be loadable by vLLM on its own:
    # copy the tokenizer/processor/generation config the grader needs.
    import glob

    aux = [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
    ]
    for ck in glob.glob(os.path.join(args.out, "checkpoint-*")) + [final]:
        for name in aux:
            src = os.path.join(args.model, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(ck, name))
        g = json.load(open(os.path.join(ck, "generation_config.json")))
        assert 106 in g["eos_token_id"], (ck, g)
        assert g.get("top_k") and g.get("top_p"), (ck, g)
    print(f"[done] aux files copied into {len(glob.glob(os.path.join(args.out, 'checkpoint-*'))) + 1} dirs", flush=True)

    gc = json.load(open(os.path.join(final, "generation_config.json")))
    assert 106 in gc["eos_token_id"], gc
    print(f"[done] saved {final}; generation_config={gc}", flush=True)


if __name__ == "__main__":
    main()
