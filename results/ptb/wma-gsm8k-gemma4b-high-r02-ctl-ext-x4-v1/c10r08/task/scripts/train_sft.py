#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt on GSM8K-style CoT.

Rows are rendered with the SAME chat template the grader passes to vLLM
(templates/gemma3.jinja), so training and grading see identical strings.
Loss is on the completion only; every completion ends with <end_of_turn>
(token 106), which is in the model's generation_config eos list and is
therefore what vLLM stops on.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmt  # noqa: E402

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")


class SFTRows(Dataset):
    def __init__(self, path, tokenizer, max_seq_len, limit=None, report=True):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        tpl = fmt.template_text()
        eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        self.examples = []
        n_trunc = 0
        lens = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                prompt = tokenizer.apply_chat_template(
                    fmt.build_messages(r["question"], r.get("system")),
                    chat_template=tpl, tokenize=False, add_generation_prompt=True)
                p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                # the target text carries the terminator, so the file itself is
                # checkable by preflight's stop_token_consistent
                assert r["target"].endswith("<end_of_turn>")
                c_ids = tokenizer(r["target"], add_special_tokens=False)["input_ids"]
                assert c_ids[-1] == eot_id, c_ids[-3:]
                total = len(p_ids) + len(c_ids)
                lens.append(total)
                if total > max_seq_len:
                    n_trunc += 1
                    continue                      # drop, never truncate a target
                self.examples.append((p_ids, c_ids))
        if report:
            lens.sort()
            print(f"[data] {path}: kept {len(self.examples)} / {len(lens)}; "
                  f"dropped-too-long {n_trunc} ({n_trunc / max(1, len(lens)):.3%}); "
                  f"len p50={lens[len(lens) // 2]} p99={lens[int(len(lens) * .99)]} "
                  f"max={lens[-1]}", flush=True)
            assert n_trunc / max(1, len(lens)) < 0.02, "more than 2% of rows exceed max_seq_len"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        p, c = self.examples[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--greedy-gen-config", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTRows(args.data, tok, args.max_seq_len, limit=args.limit)

    # ---- pitfall guards, printed so they land in the log --------------------
    p, c = ds.examples[0]
    print("[check] template sha:", fmt.template_sha(), flush=True)
    print("[check] example rendered tail:",
          repr(tok.decode(p[-12:]) + " || " + tok.decode(c[-14:])), flush=True)
    assert c[-1] == tok.convert_tokens_to_ids("<end_of_turn>")
    if args.dry_run:
        print("[dry-run] ok", flush=True)
        return

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, torch_dtype=torch.bfloat16, attn_implementation="eager")
    # text-only task: freeze the vision tower and the projector
    frozen = 0
    for name, prm in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            prm.requires_grad_(False)
            frozen += prm.numel()
    print(f"[model] frozen {frozen / 1e6:.0f}M vision params; trainable "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B", flush=True)
    model.config.use_cache = False
    # transformers validates generation_config on EVERY save (intermediate
    # checkpoints included) and rejects temperature=0.0 with do_sample=False --
    # the greedy config exp-03 adopted, which vLLM reads without complaint.
    # Neutralise it here, at load time, so no save path can trip over it; the
    # real generation_config.json is written directly after training.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid")

    steps_per_epoch = math.ceil(len(ds) / (args.bs * args.grad_accum))
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 100000,
        save_total_limit=3,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        use_liger_kernel=True,
        adam_beta2=0.95,
        group_by_length=True,
        length_column_name=None,
        dataloader_num_workers=4,
        report_to=[],
        seed=args.seed,
        save_safetensors=True,
    )
    print(f"[train] rows={len(ds)} steps/epoch={steps_per_epoch}", flush=True)

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    final = os.path.join(args.out, "final")
    # transformers validates generation_config on save and rejects
    # temperature=0.0 with do_sample=False -- which is exactly the greedy config
    # a previous card wrote into the parent, and vLLM is perfectly happy with.
    # Neutralise it here; the real file is written directly further down.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # keep the multimodal preprocessor files so vLLM loads the dir as gemma3
    import shutil
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    # generation_config decides what vLLM does at grading time (it reads this
    # file: ModelConfig.get_diff_sampling_param with generation_config="auto").
    # Default here is the parent's file verbatim, so a card that only changes
    # the decode configuration stays a separate, attributable experiment.
    if args.greedy_gen_config:
        gen = {"bos_token_id": 2, "cache_implementation": "hybrid",
               "do_sample": False, "temperature": 0.0,
               "eos_token_id": [1, 106], "pad_token_id": 0,
               "transformers_version": "4.50.0.dev0"}
    else:
        gen = json.load(open(os.path.join(BASE, "generation_config.json")))
    with open(os.path.join(final, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    print("[done] saved", final, flush=True)


if __name__ == "__main__":
    main()
