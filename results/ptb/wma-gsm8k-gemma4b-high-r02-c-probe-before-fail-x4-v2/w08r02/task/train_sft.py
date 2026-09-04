#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on a GSM8K-style corpus.

Two things here are not boilerplate, both forced by gemma-3's 262144-token
vocabulary (a fixed micro-batch OOMs in the cross-entropy, not in the model):

  * loss is computed only at positions whose label is not -100, by running
    lm_head on the selected hidden states. Mathematically identical to
    transformers' own loss (checked at startup by --check-loss), but it never
    materialises logits for prompt tokens.
  * batches are built to a token budget rather than a fixed sequence count, so
    a batch of 2400-token few-shot rows and a batch of 330-token zero-shot rows
    both use the same memory.

The vision tower and the multimodal projector are frozen; only the language
model trains. Rows are rendered with the grader's own chat template
(sft_common.py) and the target ends with <end_of_turn>, the terminator the
grading template stops on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sft_common import BASE_SNAPSHOT, TEMPLATE_SHA256, encode_row, get_tokenizer, load_jsonl


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def build_batches(rows, total_budget: int, label_budget: int, seed: int):
    """Greedy length-sorted batching under a total-token and labelled-token cap."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["length"])
    batches, cur, cur_lab = [], [], 0
    for i in order:
        n = rows[i]["length"]
        lab = rows[i]["n_label"]
        trial = len(cur) + 1
        maxlen = max((rows[j]["length"] for j in cur), default=0)
        maxlen = max(maxlen, n)
        if cur and (trial * maxlen > total_budget or cur_lab + lab > label_budget):
            batches.append(cur)
            cur, cur_lab = [i], lab
        else:
            cur.append(i)
            cur_lab += lab
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


class BatchList:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


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


def selective_loss(model, inputs, num_items_in_batch=None):
    labels = inputs["labels"]
    out = model.model(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False
    )
    h = out[0][:, :-1, :]
    y = labels[:, 1:]
    sel = y != -100
    hs = h[sel]
    ys = y[sel]
    logits = model.lm_head(hs).float()
    if num_items_in_batch is not None:
        return F.cross_entropy(logits, ys, reduction="sum") / num_items_in_batch
    return F.cross_entropy(logits, ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--parent", default=BASE_SNAPSHOT)
    ap.add_argument("--out", default="ckpts/exp-02")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--label-budget", type=int, default=5000)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=3328)
    ap.add_argument("--max-rows", type=int, default=-1)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-checkpointing", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--check-loss", type=int, default=1)
    args = ap.parse_args()

    print(f"chat template sha256 = {TEMPLATE_SHA256}", flush=True)
    tok = get_tokenizer(BASE_SNAPSHOT)

    raw = load_jsonl(args.data)
    random.Random(args.seed).shuffle(raw)
    if args.max_rows > 0:
        raw = raw[: args.max_rows]

    t0 = time.time()
    enc, dropped = [], 0
    for r in raw:
        e = encode_row(tok, r, args.max_seq_len)
        if e is None:
            dropped += 1
            continue
        enc.append(
            {
                "input_ids": e["input_ids"],
                "labels": e["labels"],
                "length": len(e["input_ids"]),
                "n_label": e["n_target"],
            }
        )
    print(
        f"encoded {len(enc)} rows in {time.time()-t0:.0f}s, dropped {dropped} "
        f"({dropped/max(len(raw),1):.3%}) over max_seq_len={args.max_seq_len}",
        flush=True,
    )
    lens = sorted(x["length"] for x in enc)
    tot = sum(lens)
    lab = sum(x["n_label"] for x in enc)
    print(
        f"len p50={lens[len(lens)//2]} p95={lens[int(.95*len(lens))]} max={lens[-1]} "
        f"tokens={tot/1e6:.1f}M labelled={lab/1e6:.1f}M",
        flush=True,
    )

    batches = build_batches(enc, args.token_budget, args.label_budget, args.seed)
    bs = [len(b) for b in batches]
    print(f"{len(batches)} micro-batches, seqs/batch min={min(bs)} p50={sorted(bs)[len(bs)//2]} max={max(bs)}", flush=True)

    from transformers import Gemma3ForConditionalGeneration, Trainer, TrainingArguments

    attn = "flash_attention_2"
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.parent, dtype=torch.bfloat16, attn_implementation=attn
        )
    except Exception as e:  # pragma: no cover
        print(f"flash_attention_2 unavailable ({e!r}); falling back to sdpa", flush=True)
        attn = "sdpa"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.parent, dtype=torch.bfloat16, attn_implementation=attn
        )
    print(f"attn_implementation={attn}", flush=True)

    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M params, trainable {trainable/1e6:.0f}M", flush=True)
    model.config.use_cache = False

    # transformers refuses to save a GenerationConfig with do_sample=False and an
    # explicit temperature, which is exactly the greedy config we ship for vLLM.
    # Keep the in-memory one valid so save_model works (and so a later card can
    # resume from this checkpoint); the greedy file is written by hand afterwards.
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid", do_sample=False,
    )

    if args.check_loss:
        # the selected-token loss must equal transformers' own loss
        coll = Collator(tok.pad_token_id)
        b = coll([enc[i] for i in batches[0][:2]])
        model.cuda()
        b = {k: v.cuda() for k, v in b.items()}
        with torch.no_grad():
            ref = model(**b).loss.item()
            mine = selective_loss(model, b).item()
        print(f"loss check: transformers={ref:.6f} selective={mine:.6f} diff={abs(ref-mine):.2e}", flush=True)
        assert abs(ref - mine) < 2e-2, "selective loss does not reproduce the reference loss"
        del b
        torch.cuda.empty_cache()

    class TokenBudgetTrainer(Trainer):
        def get_train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_sampler=BatchList(batches),
                collate_fn=self.data_collator,
                num_workers=2,
                pin_memory=True,
            )

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = selective_loss(model, inputs, num_items_in_batch)
            return (loss, None) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,  # unused: batch_sampler drives the loader
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=25,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 10**9,
        save_total_limit=3,
        report_to=[],
        gradient_checkpointing=bool(args.grad_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        seed=args.seed,
        remove_unused_columns=False,
        save_safetensors=True,
        accelerator_config={"split_batches": False},
    )

    trainer = TokenBudgetTrainer(
        model=model,
        args=targs,
        train_dataset=Rows(enc),
        data_collator=Collator(tok.pad_token_id),
    )
    out = trainer.train()
    print(out, flush=True)

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)

    # decode config: greedy. The base checkpoint ships do_sample/top_k/top_p and
    # no temperature, so vLLM serves it at t=1.0 (confirmed in exp-01); for a
    # single-sample grader greedy is the right decode.
    gen = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        # temperature must be present: vLLM's get_diff_sampling_param only adopts
        # keys that appear in generation_config's diff dict, and with none of
        # temperature/top_p/top_k present it falls back to its own t=1.0 default.
        # top_k is deliberately omitted -- top_k: -1 is a vLLM sentinel that makes
        # a later transformers save_pretrained raise.
        "do_sample": False,
        "temperature": 0.0,
        "transformers_version": "4.57.3",
    }
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    with open(os.path.join(final, "train_meta.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "template_sha256": TEMPLATE_SHA256,
                "n_rows": len(enc),
                "n_micro_batches": len(batches),
                "attn": attn,
                "log_history": trainer.state.log_history[-5:],
            },
            f,
            indent=2,
        )
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
