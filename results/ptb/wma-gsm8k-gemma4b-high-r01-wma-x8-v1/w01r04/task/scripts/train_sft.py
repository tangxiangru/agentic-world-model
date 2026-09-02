#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style CoT.

The training string is built byte-for-byte the way templates/gemma3.jinja
renders a single user turn (verified by scripts/render_check.py), so what the
model is trained on is what the grader sends. Loss is on the completion only.
"""
import argparse
import json
import os
import shutil

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

SNAP_DEFAULT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)


def build_dataset(path: str, tok, max_seq_len: int):
    rows = [json.loads(l) for l in open(path)]
    prompts = [
        f"<start_of_turn>user\n{r['prompt']}<end_of_turn>\n<start_of_turn>model\n"
        for r in rows
    ]
    comps = [r["completion"] for r in rows]
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    c_ids = tok(comps, add_special_tokens=False)["input_ids"]

    bos = tok.bos_token_id
    feats, dropped = [], 0
    for p, c in zip(p_ids, c_ids):
        ids = [bos] + p + c
        if len(ids) > max_seq_len:
            dropped += 1
            continue
        labels = [-100] * (1 + len(p)) + list(c)
        feats.append({"input_ids": ids, "labels": labels, "length": len(ids)})
    print(f"rows {len(rows)}  kept {len(feats)}  dropped(too long) {dropped}")
    assert dropped / len(rows) < 0.02, "more than 2% of rows would truncate"
    n_loss = sum(sum(1 for x in f["labels"] if x != -100) for f in feats)
    print(f"loss tokens {n_loss}  total tokens {sum(f['length'] for f in feats)}")
    assert n_loss > 0
    return Dataset.from_list(feats)


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, lab, att = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * k)
            lab.append(b["labels"] + [-100] * k)
            att.append([1] * len(b["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
        }


def fix_artifacts(out_dir: str, snap: str, tok, decode: str = "greedy") -> None:
    """Trainer collapses eos_token_id [1, 106] to 1 on save; vLLM would then
    never stop on <end_of_turn>. Restore the shipped generation_config and make
    sure the tokenizer/processor files travel with the weights.

    decode='greedy' also pins temperature 0 (exp-03: +6.0 pts on dev-150), which
    is what vLLM reads out of generation_config.json since evaluate.py forwards
    no sampling parameters."""
    gen = json.load(open(os.path.join(snap, "generation_config.json")))
    if decode == "greedy":
        gen.update({"do_sample": False, "temperature": 0.0, "top_k": 0, "top_p": 1.0})
    gen["eos_token_id"] = [1, 106]
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    tok.save_pretrained(out_dir)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(snap, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fn))
    print(f"fixed artifacts in {out_dir}: eos_token_id={gen['eos_token_id']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SNAP_DEFAULT)
    ap.add_argument("--data", default="/home/ben/task/data/sft_v1.jsonl")
    ap.add_argument("--out", default="/home/ben/task/ckpts/exp-02")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--decode", choices=["greedy", "shipped"], default="greedy")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = build_dataset(args.data, tok, args.max_seq_len)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    print(type(model).__name__)
    # A greedy generation_config (temperature 0 with do_sample False) is what we
    # want on disk, but GenerationConfig.save_pretrained rejects that combination
    # and aborts the checkpoint save. Train with the shipped config attached and
    # let fix_artifacts() write the greedy one after the weights are on disk.
    model.generation_config = GenerationConfig.from_pretrained(SNAP_DEFAULT)
    # text-only task: keep the vision tower frozen so its weights stay valid
    frozen = 0
    for name, p in model.named_parameters():
        if name.startswith(("model.vision_tower", "vision_tower",
                            "model.multi_modal_projector", "multi_modal_projector")):
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {frozen/1e6:.0f}M   trainable {trainable/1e6:.0f}M")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim=args.optim,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    fix_artifacts(final, SNAP_DEFAULT, tok, args.decode)
    for d in os.listdir(args.out):
        if d.startswith("checkpoint-"):
            fix_artifacts(os.path.join(args.out, d), SNAP_DEFAULT, tok, args.decode)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print("done ->", final)


if __name__ == "__main__":
    main()
