"""Completion-only SFT for gemma-3-4b-pt on the harness's GSM8K contract.

Rows are rendered with templates/gemma3.jinja (the grader's own file), the loss
covers only the assistant turn, and every target ends with <end_of_turn>.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from common import BASE_SNAPSHOT, STOP_TOKEN, load_tokenizer, render_prompt, template_sha

ROOT = Path(__file__).resolve().parent.parent


class SFTRows(Dataset):
    def __init__(self, path, tok, max_seq_len, report=True):
        self.rows = []
        n_skip = 0
        lens = []
        for line in Path(path).open():
            r = json.loads(line)
            prompt = render_prompt(tok, r["messages"])
            completion = r["completion"]
            assert completion.endswith(STOP_TOKEN), "target does not end with the stop token"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(completion, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(c_ids) > max_seq_len:
                n_skip += 1
                continue
            lens.append(len(p_ids) + len(c_ids))
            self.rows.append((p_ids, c_ids))
        if report:
            lens.sort()
            print(f"[data] {path}: kept {len(self.rows)}, skipped {n_skip} over "
                  f"max_seq_len={max_seq_len}; len p50={lens[len(lens)//2]} "
                  f"p95={lens[int(len(lens)*0.95)]} max={lens[-1]}", flush=True)
            assert n_skip / max(1, n_skip + len(self.rows)) < 0.02, "over 2% truncated"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(features, pad_id):
    n = max(len(f["input_ids"]) for f in features)
    input_ids, labels, mask = [], [], []
    for f in features:
        k = n - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * k)
        labels.append(f["labels"] + [-100] * k)
        mask.append([1] * len(f["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--parent", default=BASE_SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1792)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--save-steps", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"[template] templates/gemma3.jinja sha256:{template_sha()}", flush=True)
    tok = load_tokenizer(BASE_SNAPSHOT)

    ds_parts = [SFTRows(p, tok, args.max_seq_len) for p in args.data]
    rows = [r for d in ds_parts for r in d.rows]
    random.Random(args.seed).shuffle(rows)
    ds = ds_parts[0]
    ds.rows = rows
    print(f"[data] total rows {len(ds)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn)
    if hasattr(model, "vision_tower"):
        for p in model.vision_tower.parameters():
            p.requires_grad_(False)
        for p in model.multi_modal_projector.parameters():
            p.requires_grad_(False)
        print("[model] vision tower frozen", flush=True)
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
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=99,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim="adamw_torch_fused",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      processing_class=tok,
                      data_collator=lambda f: collate(f, tok.pad_token_id))
    trainer.add_callback(FinalizeCheckpoints(tok))
    trainer.train()

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    finalize(final)
    print(f"[done] saved {final}", flush=True)


class FinalizeCheckpoints(TrainerCallback):
    """Make every intermediate checkpoint independently loadable by vLLM."""

    def __init__(self, tok):
        self.tok = tok

    def on_save(self, args, state, control, **kwargs):
        d = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(d):
            self.tok.save_pretrained(d)
            finalize(d)
        return control


def finalize(out_dir: str) -> None:
    """Restore the artefacts a Trainer save drops or rewrites.

    The base generation_config lists eos_token_id [1, 106]; a Trainer save can
    collapse that to a scalar, which would leave vLLM without <end_of_turn> as a
    stop token and every completion running to the 4000-token cap.
    """
    import shutil
    src = Path(BASE_SNAPSHOT)
    dst = Path(out_dir)
    for name in ("generation_config.json", "preprocessor_config.json",
                 "processor_config.json"):
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
    gc = json.loads((dst / "generation_config.json").read_text())
    assert gc["eos_token_id"] == [1, 106], gc
    print(f"[finalize] generation_config restored from base: {gc}", flush=True)


if __name__ == "__main__":
    main()
