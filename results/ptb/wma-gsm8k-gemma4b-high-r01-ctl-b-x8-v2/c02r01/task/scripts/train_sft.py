#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt, rendered through the grader's own
templates/gemma3.jinja so training and grading see byte-identical strings.

Row -> "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
       + "{completion}" + "<end_of_turn>"
Loss is on the completion and the terminating <end_of_turn> only.
"""
import argparse, hashlib, json, os, random, sys

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoProcessor, Trainer, TrainingArguments,
                          Gemma3ForConditionalGeneration, set_seed)

TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
END_OF_TURN_ID = 106


def render_prompt(tok, template, user_content):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        chat_template=template, add_generation_prompt=True, tokenize=False,
    )


class SFTData(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            n = L - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * n)
            labels.append(f["labels"] + [-100] * n)
            mask.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def build_rows(tok, template, data_path, max_seq_len, limit=None, verbose=True):
    rows, n_trunc, n_prefix_bad, n_total = [], 0, 0, 0
    lens = []
    with open(data_path) as f:
        for line in f:
            if limit is not None and n_total >= limit:
                break
            r = json.loads(line)
            n_total += 1
            p = render_prompt(tok, template, r["prompt"])
            comp = r["completion"]
            if not comp.endswith("<end_of_turn>"):
                comp += "<end_of_turn>"
            full = p + comp
            pid = tok(p, add_special_tokens=False)["input_ids"]
            fid = tok(full, add_special_tokens=False)["input_ids"]
            if fid[: len(pid)] != pid:
                n_prefix_bad += 1
                continue
            lens.append(len(fid))
            if len(fid) > max_seq_len:
                n_trunc += 1
                continue
            labels = [-100] * len(pid) + fid[len(pid):]
            rows.append({"input_ids": fid, "labels": labels})
    if verbose:
        lens.sort()
        print(f"[data] {data_path}: kept {len(rows)}/{n_total} "
              f"(dropped {n_trunc} over max_seq_len={max_seq_len}, {n_prefix_bad} prefix mismatch)")
        if lens:
            print(f"[data] token length p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build data, print stats, exit")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--no-ckpt", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    template = open(TEMPLATE_PATH).read()
    print("[template] sha256", hashlib.sha256(template.encode()).hexdigest())

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = build_rows(tok, template, args.data, args.max_seq_len, args.limit)

    # ---- dry-run sanity: every target ends in <end_of_turn>, marker appears once
    bad_eos = sum(1 for r in rows if r["input_ids"][-1] != END_OF_TURN_ID)
    dec = tok.decode(rows[0]["input_ids"])
    n_marker = dec.count("\nANSWER: ")
    print(f"[check] rows not ending in <end_of_turn>: {bad_eos}")
    print(f"[check] first row decoded (tail 300 chars): {dec[-300:]!r}")
    print(f"[check] 'ANSWER: ' occurrences in first row: {n_marker}")
    n_lossless = sum(1 for r in rows if all(l == -100 for l in r["labels"]))
    print(f"[check] rows with zero loss tokens: {n_lossless}")
    assert bad_eos == 0 and n_lossless == 0, "target/eos defect"
    if args.dry_run:
        print("[dry-run] ok, exiting")
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    # text-only task: freeze the vision stack so it neither trains nor drifts
    frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") \
           or name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] frozen {frozen/1e6:.1f}M, trainable {trainable/1e6:.1f}M")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=10,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=4,
        save_only_model=True,
        gradient_checkpointing=(not args.no_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        report_to=[],
        seed=args.seed,
        group_by_length=True,
        dataloader_num_workers=4,
        use_liger_kernel=(not args.no_liger),
        max_grad_norm=1.0,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=SFTData(rows),
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    try:
        proc = AutoProcessor.from_pretrained(args.model)
        proc.save_pretrained(args.out)
    except Exception as e:
        print("[warn] processor save failed:", e)
    print("[done] saved to", args.out)


if __name__ == "__main__":
    main()
