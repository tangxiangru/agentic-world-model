#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style math CoT.

The prompt is rendered with the *grader's own* chat template (templates/gemma3.jinja,
hash-checked) so training and evaluation see byte-identical strings. Loss is on the
completion only, and every target ends with <end_of_turn> (token 106), which is the
terminator in the base checkpoint's generation_config eos_token_id [1, 106].
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
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
TEMPLATE_SHA = "b8b3a41ffc9e5c1c3f4dc27f1fa2b1a2"  # filled at first run; informational
EOT = 106  # <end_of_turn>


def load_template() -> str:
    with open(TEMPLATE, "rb") as f:
        raw = f.read()
    print(f"chat template {TEMPLATE} md5={hashlib.md5(raw).hexdigest()} bytes={len(raw)}")
    return raw.decode("utf-8")


class SFTData(Dataset):
    def __init__(self, path: str, tok, template: str, max_len: int, limit: int = 0,
                 pretokenize: bool = True):
        self.rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                self.rows.append(json.loads(line))
        self.tok = tok
        self.template = template
        self.max_len = max_len
        self.enc: list[dict] | None = None
        if pretokenize:
            self._pretokenize()

    def _pretokenize(self) -> None:
        import numpy as np
        prompts = [self.render_prompt(r) for r in self.rows]
        comps = [r["completion"].strip() for r in self.rows]
        B = 2000
        self.enc = []
        self.n_trunc = 0
        for i in range(0, len(prompts), B):
            pe = self.tok(prompts[i:i + B], add_special_tokens=False)["input_ids"]
            ce = self.tok(comps[i:i + B], add_special_tokens=False)["input_ids"]
            for p, c in zip(pe, ce):
                assert c[-1] == EOT, "target does not end with <end_of_turn>"
                ids = p + c
                labels = [-100] * len(p) + c
                if len(ids) > self.max_len:
                    self.n_trunc += 1
                    ids = ids[-self.max_len:]
                    labels = labels[-self.max_len:]
                self.enc.append({
                    "input_ids": np.asarray(ids, dtype=np.int32),
                    "labels": np.asarray(labels, dtype=np.int32),
                })
        print(f"pretokenized {len(self.enc)} rows; truncated {self.n_trunc}")

    def __len__(self) -> int:
        return len(self.rows)

    def render_prompt(self, r) -> str:
        msgs = []
        if r.get("system"):
            msgs.append({"role": "system", "content": r["system"]})
        msgs.append({"role": "user", "content": r["prompt"]})
        return self.tok.apply_chat_template(
            msgs, chat_template=self.template, tokenize=False, add_generation_prompt=True
        )

    def encode(self, r) -> dict:
        p = self.tok(self.render_prompt(r), add_special_tokens=False)["input_ids"]
        c = self.tok(r["completion"].strip(), add_special_tokens=False)["input_ids"]
        assert c[-1] == EOT, "target does not end with <end_of_turn>"
        ids = p + c
        labels = [-100] * len(p) + c[:]
        if len(ids) > self.max_len:  # keep the tail (the answer) rather than the head
            ids = ids[-self.max_len:]
            labels = labels[-self.max_len:]
        return {"input_ids": ids, "labels": labels}

    def __getitem__(self, i: int) -> dict:
        if self.enc is not None:
            e = self.enc[i]
            return {"input_ids": e["input_ids"].tolist(), "labels": e["labels"].tolist()}
        return self.encode(self.rows[i])


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats: list[dict]) -> dict:
        n = max(len(f["input_ids"]) for f in feats)
        ids, labs, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(list(f["input_ids"]) + [self.pad_id] * k)
            labs.append(list(f["labels"]) + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    if not args.no_liger and not args.dry_run:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger kernel applied to gemma3 (fused linear cross entropy)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    template = load_template()
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTData(args.data, tok, template, args.max_seq_len, args.limit,
                 pretokenize=not args.dry_run)
    print(f"rows: {len(ds)}")

    # ---- dry run: prove train-time and grade-time renderings agree -------------
    ex = ds.encode(ds.rows[0])
    full = tok.apply_chat_template(
        ([{"role": "system", "content": ds.rows[0]["system"]}] if ds.rows[0].get("system") else [])
        + [{"role": "user", "content": ds.rows[0]["prompt"]},
           {"role": "assistant",
            "content": ds.rows[0]["completion"].replace("<end_of_turn>", "")}],
        chat_template=template, tokenize=False, add_generation_prompt=False,
    )
    ours = tok.decode(ex["input_ids"])
    assert full.rstrip("\n") == ours, f"RENDER MISMATCH\n---full---\n{full!r}\n---ours---\n{ours!r}"
    assert ex["input_ids"][-1] == EOT, "target does not end in <end_of_turn>"
    assert ex["labels"][-1] == EOT and ex["labels"][0] == -100
    n_lab = sum(1 for x in ex["labels"] if x != -100)
    print(f"dry-run ok: {len(ex['input_ids'])} tokens, {n_lab} loss tokens")
    print("---- rendered example ----")
    print(ours[:1200])
    print("--------------------------")
    if args.dry_run:
        lens = [len(ds.encode(r)["input_ids"]) for r in ds.rows[: min(3000, len(ds.rows))]]
        over = sum(1 for x in lens if x >= args.max_seq_len)
        print(f"len p50={sorted(lens)[len(lens)//2]} max={max(lens)} "
              f"truncated={over}/{len(lens)} ({100*over/len(lens):.2f}%)")
        return

    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.parent, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print("flash_attention_2 unavailable, falling back to eager:", e)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.parent, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
    model.config.use_cache = False
    # A parent trained by this script ships generation_config {temperature: 0.0} with no
    # do_sample, which GenerationConfig.save_pretrained rejects (ValueError) - so every
    # Trainer checkpoint save would abort mid-run. Reset the in-memory config to the base
    # model's valid one; the greedy config is written back over final/ at the end, by
    # json.dump, which does not go through that validation.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig.from_pretrained(BASE)
    n_froz = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_froz += p.numel()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e9:.2f}B, frozen {n_froz/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 999999,
        save_total_limit=2,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        adam_beta2=0.95,
        max_grad_norm=1.0,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=8,
        seed=args.seed,
        report_to=[],
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id)
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(BASE).save_pretrained(final)
    except Exception as e:  # processor is only needed for images; not fatal
        print("processor save skipped:", e)
    # vLLM builds its default SamplingParams from generation_config.json and ignores
    # `do_sample`; dropping top_k/top_p while leaving no temperature yields *unrestricted*
    # t=1.0 sampling (exp-02: base model emitted unicode garbage). Greedy must be asked
    # for by name: temperature 0.
    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    for k in ("do_sample", "top_k", "top_p"):
        gc.pop(k, None)
    gc["temperature"] = 0.0
    json.dump(gc, open(os.path.join(final, "generation_config.json"), "w"), indent=2)
    print("saved", final)


if __name__ == "__main__":
    main()
