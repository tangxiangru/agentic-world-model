#!/usr/bin/env python3
"""Full fine-tuning SFT of gemma-3-4b-pt for BFCL-style function calling.

- Renders each example with templates/gemma3_tool_calling.jinja (the exact
  template the grader uses) so train/eval formats match byte-for-byte.
- Completion-only loss (prompt tokens masked via token-level common prefix).
- Vision tower + multimodal projector frozen; language model trained.
"""
import argparse, json, os
import torch
from transformers import (AutoTokenizer, AutoModelForImageTextToText,
                          Trainer, TrainingArguments)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "templates/gemma3_tool_calling.jinja"


def build_examples(tok, path, max_seq_len):
    rows = [json.loads(l) for l in open(path)]
    out = []
    n_trunc = 0
    for row in rows:
        msgs = [
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": "",
             "tool_calls": [{"type": "function",
                             "function": {"name": row["name"], "arguments": row["arguments"]}}]},
        ]
        full = tok.apply_chat_template(msgs, tools=row["tools"], tokenize=False,
                                       add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs[:-1], tools=row["tools"], tokenize=False,
                                         add_generation_prompt=True)
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        # token-level common prefix = completion boundary
        n = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            n += 1
        if len(full_ids) > max_seq_len:
            n_trunc += 1
            continue
        labels = [-100] * n + full_ids[n:]
        out.append({"input_ids": full_ids, "labels": labels,
                    "attention_mask": [1] * len(full_ids)})
    print(f"built {len(out)} examples ({n_trunc} dropped for >{max_seq_len} tokens)")
    return out


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parent", default=SNAP)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template = open(TEMPLATE).read()

    ds = build_examples(tok, args.data, args.max_seq_len)

    model = AutoModelForImageTextToText.from_pretrained(
        args.parent, torch_dtype=torch.bfloat16, attn_implementation="eager")
    # freeze vision tower + projector; train language model only
    frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            frozen += p.numel()
    print(f"frozen params: {frozen/1e6:.1f}M")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

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
        logging_steps=20,
        save_strategy="no",
        seed=args.seed,
        report_to=[],
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id or 0))
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    # multimodal arch: vLLM needs the image processor configs to load the model
    import shutil
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
