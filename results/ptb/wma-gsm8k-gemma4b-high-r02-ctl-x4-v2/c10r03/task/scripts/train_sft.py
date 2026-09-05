"""Full-parameter SFT of gemma-3-4b-pt on GSM8K-style chain-of-thought data.

Rendering is done with templates/gemma3.jinja - the exact file evaluate.py hands
to vLLM - so training and grading see byte-identical prompts. Loss is on the
model turn only. Every target ends with <end_of_turn> (token 106), which is in
the model's eos list, so the graded completion terminates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from promptlib import STOP_TOKEN, fewshot_system_message, render  # noqa: E402


class BatchedRows(Dataset):
    """One item = one micro-batch, packed to a token budget.

    gemma-3's vocab is 262144, so a micro-batch is bounded by padded *tokens*,
    not by row count: a fixed batch size that is safe for the 2500-token
    few-shot rows wastes most of the GPU on the 320-token zero-shot ones.
    """

    def __init__(self, rows, token_budget: int, max_rows: int, seed: int):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
        batches, cur, curmax = [], [], 0
        for i in order:
            L = len(rows[i]["input_ids"])
            m = max(curmax, L)
            if cur and (m * (len(cur) + 1) > token_budget or len(cur) >= max_rows):
                batches.append(cur)
                cur, curmax = [], 0
                m = L
            cur.append(rows[i])
            curmax = m
        if cur:
            batches.append(cur)
        random.Random(seed).shuffle(batches)
        self.batches = batches
        tot = sum(len(b) for b in batches)
        pad = sum(len(b) * max(len(r["input_ids"]) for r in b) for b in batches)
        print(
            f"micro-batches={len(batches)} rows={tot} "
            f"avg_rows/batch={tot/len(batches):.1f} padded_tokens={pad/1e6:.1f}M",
            flush=True,
        )

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]


def collate(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def build_rows(tok, data_path, fewshot_frac, max_seq_len, seed, limit=None):
    rng = random.Random(seed)
    sys_msg = fewshot_system_message() if fewshot_frac > 0 else None

    raw = []
    with open(data_path) as f:
        for line in f:
            raw.append(json.loads(line))
    if limit:
        raw = raw[:limit]

    rows, n_trunc, lens = [], 0, []
    for r in raw:
        use_fs = rng.random() < fewshot_frac
        prompt_text, full_text = render(
            tok, r["question"], sys_msg if use_fs else None, r["target"]
        )
        # full_text = prompt + target + STOP_TOKEN + "\n"; drop the trailing newline
        full_text = full_text[: -len("\n")]
        p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        if f_ids[: len(p_ids)] != p_ids:
            raise RuntimeError("prompt is not a token-prefix of the full text")
        if f_ids[-1] != tok.convert_tokens_to_ids(STOP_TOKEN):
            raise RuntimeError(f"row does not end with {STOP_TOKEN}")
        lens.append(len(f_ids))
        if len(f_ids) > max_seq_len:
            n_trunc += 1
            continue
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        rows.append({"input_ids": f_ids, "labels": labels})

    lens.sort()
    print(
        f"rows={len(rows)} dropped_too_long={n_trunc} "
        f"({n_trunc / max(1, len(raw)):.3%})  "
        f"len p50={lens[len(lens) // 2]} p99={lens[int(len(lens) * 0.99)]} max={lens[-1]}",
        flush=True,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    ap.add_argument("--data", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--out", default="/home/ben/task/ckpts/exp-02")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--max-rows-per-batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2600)
    ap.add_argument("--fewshot-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--param-dtype", default="fp32")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = build_rows(
        tok, args.data, args.fewshot_frac, args.max_seq_len, args.seed, args.limit
    )
    ds = BatchedRows(rows, args.token_budget, args.max_rows_per_batch, args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32 if args.param_dtype == "fp32" else torch.bfloat16,
        attn_implementation=args.attn,
    )
    # text-only task: the SigLIP tower and the projector never see a gradient
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_train/1e9:.2f}B  frozen {n_frozen/1e6:.0f}M", flush=True)

    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,  # one item = one packed micro-batch
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        use_liger_kernel=True,
        max_steps=args.max_steps if args.max_steps else -1,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b[0], tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    os.makedirs(final, exist_ok=True)
    model.config.use_cache = True
    trainer.model.to(torch.bfloat16)
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    # keep the processor sidecar files so vLLM loads the multimodal config
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.model, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    # greedy decoding: vLLM reads these as its default sampling params
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(
            {
                "bos_token_id": 2,
                "eos_token_id": [1, 106],
                "pad_token_id": 0,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "cache_implementation": "hybrid",
            },
            f,
            indent=2,
        )
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
