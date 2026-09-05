#!/usr/bin/env python3
"""Full SFT of google/gemma-3-4b-pt (base snapshot) on GSM8K-train, eval format.

- Loads the immutable base snapshot as the parent (weights cannot drift).
- Renders every row with the SAME chat template the grader uses (templates/gemma3.jinja),
  so the training string == eval string (minus the 10-shot system prefix).
- Completion-only loss: prompt tokens masked to -100; target ends with <end_of_turn> (id 106),
  which is in generation_config eos [1,106] -> the model learns to stop where the grader stops.
- Vision tower / projector frozen (kept identical to base; text-only task).
"""
import argparse, json, os

import torch
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = "templates/gemma3.jinja"


def build_examples(tok, path, max_seq_len):
    rows = [json.loads(l) for l in open(path)]
    kept, truncated, lengths = [], 0, []
    for r in rows:
        msgs = r["messages"]
        # prompt = everything up to (but not including) the final assistant turn;
        # handles both [user, assistant] and [system, user, assistant].
        prompt_ids = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
        full_ids = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True)
        assert full_ids[: len(prompt_ids)] == prompt_ids, "prompt not a prefix of full render"
        lengths.append(len(full_ids))
        if len(full_ids) > max_seq_len:
            truncated += 1
            continue  # drop over-length rows rather than truncate the completion
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        kept.append({"input_ids": full_ids, "labels": labels})
    lengths.sort()
    p50 = lengths[len(lengths) // 2]
    p99 = lengths[int(len(lengths) * 0.99)]
    print(f"[data] rows={len(rows)} kept={len(kept)} dropped_overlen={truncated} "
          f"p50={p50} p99={p99} max={lengths[-1]} (max_seq_len={max_seq_len})")
    return kept


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = f["input_ids"]; lab = f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_messages.jsonl")
    ap.add_argument("--parent", default=SNAP, help="parent checkpoint to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_strategy", default="epoch")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of rows")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template = open(TEMPLATE_PATH).read()
    if tok.pad_token_id is None:
        tok.pad_token = "<pad>"

    data = build_examples(tok, args.data, args.max_seq_len)
    if args.limit:
        data = data[: args.limit]

    print(f"[model] loading parent: {args.parent}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    # freeze non-text params (vision tower + projector); train language model only
    ntrain = 0
    for n, p in model.named_parameters():
        if ("vision" in n) or ("multi_modal" in n):
            p.requires_grad_(False)
        if p.requires_grad:
            ntrain += p.numel()
    print(f"[model] trainable params ~ {ntrain/1e9:.2f}B")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_total_limit=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=data,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()
    # ensure a final, fully-loadable model dir
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print(f"[done] saved to {final}")


if __name__ == "__main__":
    main()
