#!/usr/bin/env python3
"""Completion-only SFT for google/gemma-3-4b-pt on a GSM8K-shaped corpus.

The prompt/target pair is rendered with the SAME string the grader's
templates/gemma3.jinja produces, so training and grading see one format:

    <bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{target}<end_of_turn>\n

Loss is taken on the target tokens only. The rendering is asserted against the
grader's own jinja template at startup (template_unreachable pitfall).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

BOS = "<bos>"
EOT = "<end_of_turn>"
USER_OPEN = "<start_of_turn>user\n"
MODEL_OPEN = "<start_of_turn>model\n"


def render_prompt(prompt: str) -> str:
    return f"{BOS}{USER_OPEN}{prompt.strip()}{EOT}\n{MODEL_OPEN}"


def render_target(target: str) -> str:
    """The corpus already carries the <end_of_turn> terminator; add only the newline
    the jinja template emits after it. Anything else would double the stop token."""
    t = target.strip()
    if not t.endswith(EOT):
        t += EOT
    return f"{t}\n"


def verify_template(tokenizer, template_path: str) -> None:
    """Render one example through the grader's jinja and assert we match it."""
    from jinja2 import Environment

    with open(template_path) as f:
        src = f.read()
    env = Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    tpl = env.from_string(src)
    msgs = [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]
    theirs = tpl.render(messages=msgs, bos_token=BOS, add_generation_prompt=False)
    ours = render_prompt("Q?") + render_target("A.")
    assert theirs == ours, f"template mismatch:\n{theirs!r}\n{ours!r}"

    gen = tpl.render(messages=msgs[:1], bos_token=BOS, add_generation_prompt=True)
    assert gen == render_prompt("Q?"), f"gen-prompt mismatch:\n{gen!r}\n{render_prompt('Q?')!r}"
    print("[template] byte-for-byte match with", template_path)


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int):
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p = tokenizer(render_prompt(r["prompt"]), add_special_tokens=False).input_ids
                c = tokenizer(render_target(r["completion"]), add_special_tokens=False).input_ids
                if len(p) + len(c) > max_seq_len:
                    n_trunc += 1
                    continue  # drop rather than truncate: a truncated target loses its stop token
                self.rows.append((p, c))
        print(f"[data] {len(self.rows)} rows kept, {n_trunc} dropped for exceeding max_seq_len={max_seq_len}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        return {"input_ids": p + c, "labels": [-100] * len(p) + c}

    def lengths(self):
        return [len(p) + len(c) for p, c in self.rows]


@dataclass
class Collator:
    pad_id: int
    pad_to_multiple_of: int = 16

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = ((n + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2688)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", type=str, default="adamw_bnb_8bit")
    ap.add_argument("--save-strategy", type=str, default="epoch")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--template", type=str, default="templates/gemma3.jinja")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--liger", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    verify_template(tokenizer, args.template)

    ds = SFTDataset(args.data, tokenizer, args.max_seq_len)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="flash_attention_2"
    )
    if args.liger:
        # gemma3's 262k vocab makes the materialised logit tensor the memory bottleneck
        # (bs8 x 2688 x 262144 in fp32 = 18 GB, which is what OOMed the first smoke run);
        # liger's fused linear cross-entropy never materialises it
        from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True, model=model)
        print("[liger] fused linear cross-entropy enabled")
    # text-only corpus: the vision tower and projector see no gradient signal, so
    # freezing them saves ~0.4B params of grad + optimizer state
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params {trainable/1e9:.2f}B of {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_total_limit=4,
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim=args.optim,
        max_grad_norm=1.0,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    class LenTrainer(Trainer):
        def _get_train_sampler(self, *a, **kw):
            # group_by_length needs lengths; our dataset is not a datasets.Dataset
            from transformers.trainer_pt_utils import LengthGroupedSampler

            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=ds.lengths(),
                generator=torch.Generator().manual_seed(self.args.seed),
            )

    trainer = LenTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(pad_id=tokenizer.pad_token_id or 0),
    )
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))
    tokenizer.save_pretrained(os.path.join(args.out, "final"))
    print("[done] saved", os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
