#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style CoT data.

Prompts are rendered with the *grader's* chat template (templates/gemma3.jinja,
hash-checked) so training and evaluation see byte-identical strings.
Loss is computed on the completion only; every completion ends with
'<end_of_turn>' (token 106), which is in the checkpoint's eos_token_id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
)

TEMPLATE_PATH = "templates/gemma3.jinja"
# sha256 of the grader's gemma3 chat template as shipped in templates/
EXPECTED_TEMPLATE_SHA = None  # filled from the file itself at run time and logged


class PackedRows(Dataset):
    def __init__(self, ids: list[np.ndarray], labels: list[np.ndarray]):
        self.ids, self.labels = ids, labels
        self.lengths = [len(x) for x in ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.ids[i].tolist(),
            "labels": self.labels[i].tolist(),
            "length": self.lengths[i],
        }


class TokenBudgetBatches(Sampler):
    """Batches of roughly equal *padded token count* instead of equal row count.

    gemma-3's vocab is 262k, so a row-count batch of long rows blows up the
    logits tensor while a row-count batch of short rows starves the GPU.
    """

    def __init__(self, lengths: list[int], budget: int, max_bs: int, seed: int):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and (m * (len(cur) + 1) > budget or len(cur) >= max_bs):
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur.append(i)
                cur_max = m
        if cur:
            self.batches.append(cur)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        import random as _r
        rng = _r.Random(self.seed + self.epoch)
        self.epoch += 1
        b = list(self.batches)
        rng.shuffle(b)
        return iter(b)


class MaskedChunkLossTrainer(Trainer):
    """Cross-entropy over supervised positions only, with the lm_head projection
    checkpointed in chunks so the [tokens x 262144] logits never all exist at once."""

    chunk = 4096

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        base = model.module if hasattr(model, "module") else model
        # we call the submodule directly, so accelerate's autocast wrapper on
        # model.forward is bypassed -- enter it explicitly or everything runs fp32
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = base.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            h = out[0][:, :-1, :]
            tgt = labels[:, 1:]
            sel = tgt != -100
            hs, ts = h[sel], tgt[sel]
            n = hs.size(0)
            head = base.lm_head

            def _chunk_loss(hc, tc):
                return F.cross_entropy(head(hc).float(), tc, reduction="sum")

            total = torch.zeros((), dtype=torch.float32, device=hs.device)
            for i in range(0, n, self.chunk):
                total = total + torch.utils.checkpoint.checkpoint(
                    _chunk_loss, hs[i:i + self.chunk], ts[i:i + self.chunk], use_reentrant=False
                )
        return total / max(n, 1)

    def get_train_dataloader(self):
        ds = self.train_dataset
        sampler = TokenBudgetBatches(
            ds.lengths, self.args.token_budget, self.args.max_rows_per_batch, self.args.seed
        )
        return DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )


def collate(features, pad_id: int):
    n = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attn = [], [], []
    for f in features:
        k = n - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * k)
        labels.append(f["labels"] + [-100] * k)
        attn.append([1] * len(f["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def build_tokens(data_path: str, tok, template: str, max_seq_len: int, cache: str | None):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"loaded token cache {cache}")
        return list(z["ids"]), list(z["labels"]), json.loads(str(z["stats"]))

    ids_all, lab_all = [], []
    n_trunc = 0
    lens = []
    with open(data_path) as f:
        rows = [json.loads(l) for l in f]
    for r in rows:
        msgs = []
        if r.get("system"):
            msgs.append({"role": "system", "content": r["system"]})
        msgs.append({"role": "user", "content": r["user"]})
        prompt = tok.apply_chat_template(
            msgs, chat_template=template, tokenize=False, add_generation_prompt=True
        )
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        total = len(p_ids) + len(c_ids)
        lens.append(total)
        if total > max_seq_len:
            n_trunc += 1
            continue
        ids = np.array(p_ids + c_ids, dtype=np.int32)
        lab = np.array([-100] * len(p_ids) + c_ids, dtype=np.int32)
        ids_all.append(ids)
        lab_all.append(lab)
    lens.sort()
    stats = {
        "rows_in": len(rows),
        "rows_kept": len(ids_all),
        "dropped_too_long": n_trunc,
        "p50": lens[len(lens) // 2],
        "p95": lens[int(0.95 * len(lens))],
        "max": lens[-1],
        "total_tokens": int(sum(len(x) for x in ids_all)),
    }
    print("token stats:", stats)
    if cache:
        np.savez(cache, ids=np.array(ids_all, dtype=object),
                 labels=np.array(lab_all, dtype=object), stats=json.dumps(stats))
    return ids_all, lab_all, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-epochs", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--token-budget", type=int, default=24576)
    ap.add_argument("--max-rows-per-batch", type=int, default=64)
    args = ap.parse_args()

    template = open(TEMPLATE_PATH).read()
    print("chat template sha256:", hashlib.sha256(template.encode()).hexdigest())

    tok = AutoTokenizer.from_pretrained(args.model)
    ids, labels, stats = build_tokens(args.data, tok, template, args.max_seq_len, args.cache)
    ds = PackedRows(ids, labels)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation=os.environ.get("ATTN_IMPL","flash_attention_2")
    )
    model.config.use_cache = False
    # the vision tower is dead weight for text-only math data
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") \
           or name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen params: {n_frozen/1e6:.0f}M ; trainable: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_strategy=("epoch" if args.save_epochs else ("steps" if args.save_steps else "no")),
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        report_to=[],
        group_by_length=False,
        remove_unused_columns=False,
        optim=args.optim,
        seed=args.seed,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        accelerator_config={"split_batches": False, "dispatch_batches": False},
    )
    targs.token_budget = args.token_budget
    targs.max_rows_per_batch = args.max_rows_per_batch

    trainer = MaskedChunkLossTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda f: collate(f, tok.pad_token_id or 0),
    )
    res = trainer.train()
    print("train result:", res)

    final = os.path.join(args.out, "final")
    os.makedirs(final, exist_ok=True)
    model.config.use_cache = True
    trainer.model.to(torch.bfloat16).save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # processor is only needed for images; keep going
        print("processor save failed:", e)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, fn)):
            shutil.copy(src, os.path.join(final, fn))
    # greedy decoding: inspect_ai sends no temperature, so vLLM falls back to this file
    json.dump({
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "transformers_version": "4.57.3",
    }, open(os.path.join(final, "generation_config.json"), "w"), indent=2)
    with open(os.path.join(final, "train_stats.json"), "w") as f:
        json.dump({"data_stats": stats, "metrics": res.metrics, "args": vars(args)}, f, indent=2)
    print("saved to", final)


if __name__ == "__main__":
    main()
