#!/usr/bin/env python3
"""Full-parameter SFT of google/gemma-3-4b-pt on gsm8k-style CoT targets.

Rendering uses templates/gemma3.jinja -- the byte-for-byte file the grader
passes to vLLM -- so the training string and the graded string agree
(pitfall: template_unreachable). Targets end with <end_of_turn>, which is in
the checkpoint's generation_config eos_token_id list [1, 106]
(pitfall: eos_mismatch).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

TEMPLATE_PATH = "templates/gemma3.jinja"
END_OF_TURN = "<end_of_turn>"


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-rows", type=int, default=-1)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--ga", type=int, default=4)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--no-grad-ckpt", action="store_true")
    return p.parse_args()


def load_template() -> str:
    tpl = open(TEMPLATE_PATH).read()
    print(f"template sha256 {hashlib.sha256(tpl.encode()).hexdigest()}", flush=True)
    return tpl


def main() -> None:
    a = build_args()
    torch.manual_seed(a.seed)
    tpl = load_template()
    tok = AutoTokenizer.from_pretrained(a.model)

    rows = [json.loads(l) for l in open(a.data)]
    if a.max_rows > 0:
        rows = rows[: a.max_rows]
    print(f"{len(rows)} rows", flush=True)

    eot_id = tok.convert_tokens_to_ids(END_OF_TURN)
    assert eot_id == 106, eot_id

    def render_prompt(prompt: str, system: str | None = None) -> str:
        msgs = [{"role": "user", "content": prompt}]
        if system:
            msgs.insert(0, {"role": "system", "content": system})
        return tok.apply_chat_template(
            msgs,
            chat_template=tpl,
            tokenize=False,
            add_generation_prompt=True,
        )

    def target_text(completion: str) -> str:
        c = completion.strip()
        return c if c.endswith(END_OF_TURN) else c + END_OF_TURN

    print("--- rendered example (repr) ---", flush=True)
    print(repr(render_prompt(rows[0]["prompt"])
               + target_text(rows[0]["completion"]))[:1400], flush=True)

    def encode(batch: dict) -> dict:
        out_ids, out_labels, lengths = [], [], []
        systems = batch.get("system") or [None] * len(batch["prompt"])
        for prompt, completion, system in zip(batch["prompt"],
                                              batch["completion"], systems):
            p_ids = tok(render_prompt(prompt, system),
                        add_special_tokens=False)["input_ids"]
            c_ids = tok(target_text(completion),
                        add_special_tokens=False)["input_ids"]
            ids = p_ids + c_ids
            out_ids.append(ids)
            out_labels.append([-100] * len(p_ids) + c_ids)
            lengths.append(len(ids))
        return {"input_ids": out_ids, "labels": out_labels, "length": lengths}

    ds = Dataset.from_list(rows).map(
        encode, batched=True, batch_size=1000, num_proc=16,
        remove_columns=list(rows[0].keys()),
    )

    lens = ds["length"]
    lens_sorted = sorted(lens)
    n_trunc = sum(1 for x in lens if x > a.max_seq_len)
    print(f"token length p50={lens_sorted[len(lens)//2]} "
          f"p99={lens_sorted[int(len(lens)*0.99)]} max={lens_sorted[-1]} "
          f"over_max_seq_len={n_trunc} ({n_trunc/len(lens):.4%})", flush=True)

    # never truncate: drop the overlong rows instead (pitfall seq_len_truncation)
    ds = ds.filter(lambda x: x["length"] <= a.max_seq_len, num_proc=16)
    total_tokens = sum(ds["length"])
    print(f"kept {len(ds)} rows, {total_tokens/1e6:.1f}M tokens", flush=True)

    # last label token must be the terminator the grader stops on
    for i in range(min(200, len(ds))):
        assert ds[i]["labels"][-1] == eot_id

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def collate(feats: list[dict]) -> dict:
        m = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * n)
            labels.append(f["labels"] + [-100] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    model = Gemma3ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation=a.attn,
    )
    model.config.use_cache = False
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith(
                "model.multi_modal_projector"):
            param.requires_grad = False
            frozen += param.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {trainable/1e9:.2f}B, frozen {frozen/1e6:.0f}M", flush=True)

    targs = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.ga,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=a.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps" if a.save_steps else "no",
        save_steps=a.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=not a.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim="adamw_torch_fused",
        seed=a.seed,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collate)
    trainer.train()

    final = os.path.join(a.out, "final")
    model.config.use_cache = True
    trainer.save_model(final)
    tok.save_pretrained(final)
    for f in ("generation_config.json", "preprocessor_config.json",
              "processor_config.json"):
        src = os.path.join(a.model, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(final, f)):
            shutil.copy(src, os.path.join(final, f))
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
