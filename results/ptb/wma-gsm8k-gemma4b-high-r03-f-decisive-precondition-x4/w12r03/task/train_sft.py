"""Full-parameter SFT of gemma-3-4b-pt on GSM8K-style CoT data.

Prompt/target strings are rendered with the same chat template the grader hands
to vLLM (templates/gemma3.jinja); every target ends with the template's own
terminator <end_of_turn> so generation stops right after the ANSWER line.
Loss is computed on completion tokens only.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import harness_format as hf


# greedy decoding for the grader: vLLM reads generation_config.json from the
# model dir, and gemma's shipped one turns on sampling (do_sample, top_k 64,
# top_p 0.95). Every checkpoint we might evaluate must carry this instead.
GEN_CONFIG = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "cache_implementation": "hybrid",
    "transformers_version": "4.57.3",
}


def write_gen_config(d):
    with open(os.path.join(d, "generation_config.json"), "w") as f:
        json.dump(GEN_CONFIG, f, indent=2)


class GenConfigCallback(TrainerCallback):
    """Overwrite generation_config.json in every intermediate checkpoint dir."""

    def on_save(self, args, state, control, **kwargs):
        d = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(d):
            write_gen_config(d)
            print("wrote greedy generation_config to", d)


class Rows(Dataset):
    def __init__(self, feats):
        self.feats = feats

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]


def collate(batch, pad_id):
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


def build_features(path, tok, max_seq_len, n_fewshot_rows, seed, limit_rows=0):
    rows = [json.loads(l) for l in open(path)]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit_rows:
        rows = rows[:limit_rows]
    prefix = hf.fewshot_prefix() if n_fewshot_rows else None

    feats, dropped, lens = [], 0, []
    for i, r in enumerate(rows):
        use_prefix = prefix if i < n_fewshot_rows else None
        question = r.get("question")
        prompt_text = hf.render_prompt(tok, question, prefix=use_prefix)
        target_text = r["completion"]
        if not target_text.endswith(hf.STOP_TOKEN):
            target_text += hf.STOP_TOKEN
        p = tok(prompt_text, add_special_tokens=False)["input_ids"]
        t = tok(target_text, add_special_tokens=False)["input_ids"]
        lens.append(len(p) + len(t))
        if len(p) + len(t) > max_seq_len:
            dropped += 1
            continue
        feats.append({"input_ids": p + t, "labels": [-100] * len(p) + t})
    lens = np.array(lens)
    print(
        f"rows={len(rows)} kept={len(feats)} dropped={dropped} "
        f"({dropped / len(rows):.3%}) len p50={np.percentile(lens, 50):.0f} "
        f"p99={np.percentile(lens, 99):.0f} max={lens.max()}"
    )
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--fewshot-rows", type=int, default=1000,
                    help="first N rows are rendered with the exact 10-shot eval prefix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--limit-rows", type=int, default=0, help="use only the first N rows (smoke runs)")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    feats = build_features(args.data, tok, args.max_seq_len, args.fewshot_rows,
                           args.seed, args.limit_rows)
    ds = Rows(feats)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    # A parent checkpoint saved by this script carries do_sample=False with
    # temperature 0.0 / top_k 0, which GenerationConfig.save_pretrained rejects
    # (strict validation) - it would crash the first Trainer save. Neutralise the
    # in-memory copy; write_gen_config re-writes the greedy JSON after every save.
    for _f in ("temperature", "top_k", "top_p"):
        setattr(model.generation_config, _f, None)
    # text-only training: keep the vision stack frozen (and out of the optimizer)
    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {frozen / 1e6:.0f}M params, trainable {trainable / 1e9:.2f}B")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        use_liger_kernel=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
        callbacks=[GenConfigCallback()],
    )
    trainer.train()
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    trainer.model.to(torch.bfloat16)
    trainer.model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(final)
    except Exception as e:  # processor is only needed for images, keep going
        print("processor save failed:", e)
    write_gen_config(final)
    print("saved", final)


if __name__ == "__main__":
    main()
