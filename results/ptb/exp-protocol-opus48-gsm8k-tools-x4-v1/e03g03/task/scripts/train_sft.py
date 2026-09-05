#!/usr/bin/env python3
"""Full SFT of google/gemma-3-4b-pt on GSM8K-style CoT data.

- Loads the immutable snapshot as Gemma3ForConditionalGeneration (bf16).
- Freezes the vision tower (text-only task) to save memory.
- Completion-only loss: prompt tokens masked with -100; completion ends with
  <end_of_turn> (id 106), the token the grader stops on.
- Saves through the protocol GenerationSaveContract / SaveSafeTrainer.
- Writes a greedy generation_config.json and copies processor configs so the
  final dir loads in vLLM exactly like the base snapshot.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

import torch
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)
from awm.exp_protocol.save_trainer import SaveSafeTrainer
from awm.exp_protocol.save_contract import GenerationSaveContract

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
EOT_ID = 106  # <end_of_turn>


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build_dataset(data_path, tok, max_len):
    ids_list, labels_list = [], []
    n_trunc = 0
    for line in open(data_path):
        r = json.loads(line)
        p = tok.encode(r["prompt"], add_special_tokens=False)
        c = tok.encode(r["completion"], add_special_tokens=False)
        ids = p + c
        labels = [-100] * len(p) + list(c)
        if len(ids) > max_len:
            n_trunc += 1
            ids = ids[:max_len]
            labels = labels[:max_len]
        assert ids[-1] == EOT_ID or len(p) + len(c) > max_len, "target must end in EOT"
        ids_list.append(ids)
        labels_list.append(labels)
    print(f"dataset: {len(ids_list)} rows, {n_trunc} truncated", flush=True)
    return ids_list, labels_list


class Collator:
    def __init__(self, pad_id, pad_to_multiple_of=8):
        self.pad_id = pad_id
        self.m = pad_to_multiple_of

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        if self.m:
            maxlen = ((maxlen + self.m - 1) // self.m) * self.m
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = f["input_ids"]; lab = f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class DS(torch.utils.data.Dataset):
    def __init__(self, ids, labels):
        self.ids = ids; self.labels = labels
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return {"input_ids": self.ids[i], "labels": self.labels[i]}


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.parent)

    ids, labels = build_dataset(args.data, tok, args.max_len)
    train_ds = DS(ids, labels)

    print("loading model...", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, torch_dtype=torch.bfloat16, attn_implementation="eager")
    # freeze vision tower + multimodal projector (text-only task)
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision" in name or "multi_modal" in name or "mm_" in name:
            p.requires_grad_(False); n_frozen += p.numel()
    print(f"frozen params: {n_frozen/1e6:.1f}M", flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    saves = GenerationSaveContract(policy="inactive_sampling_v1")
    report = saves.check_before_compute(model)
    print("save-contract precheck:", report, flush=True)

    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = SaveSafeTrainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=Collator(tok.pad_token_id or 0),
        generation_save_contract=saves,
    )
    saves.check_before_compute(trainer.model)
    trainer.train()

    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    print("saving final model...", flush=True)
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))

    # copy processor configs so vLLM loads the multimodal model like the base
    for fn in ["preprocessor_config.json", "processor_config.json", "added_tokens.json"]:
        src = Path(args.parent) / fn
        if src.exists():
            shutil.copy(str(src), str(final / fn))
    # greedy generation config (deterministic, best for math)
    gen = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
           "cache_implementation": "hybrid", "do_sample": False}
    with open(final / "generation_config.json", "w") as f:
        json.dump(gen, f, indent=2)

    # verify essential files
    need = ["config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json"]
    missing = [n for n in need if not (final / n).exists()]
    weights = list(final.glob("*.safetensors"))
    print("final dir:", sorted(os.listdir(final)), flush=True)
    print("missing:", missing, "weight shards:", len(weights), flush=True)
    assert not missing and weights, "final model incomplete"
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
