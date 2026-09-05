#!/usr/bin/env python3
import argparse, json, os, math, random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, set_seed)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, limit=None, prompt_key="prompt", resp_key="response"):
        self.ex = []
        skipped = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and len(self.ex) >= limit:
                    break
                d = json.loads(line)
                p = "<bos><start_of_turn>user\n" + d[prompt_key].strip() + "<end_of_turn>\n<start_of_turn>model\n"
                r = d[resp_key].strip() + "<end_of_turn>\n"
                pi = tok(p, add_special_tokens=False)["input_ids"]
                ri = tok(r, add_special_tokens=False)["input_ids"]
                if len(pi) + len(ri) > max_len:
                    skipped += 1
                    continue
                self.ex.append((pi, ri))
        print(f"loaded {len(self.ex)} examples, skipped {skipped} too long", flush=True)
        self.lengths = [len(a) + len(b) for a, b in self.ex]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        pi, ri = self.ex[i]
        ids = pi + ri
        labels = [-100] * len(pi) + list(ri)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        m = int(math.ceil(m / 16) * 16)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            pad = m - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--out", default="runs/sft_v1")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--liger", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTData(args.data, tok, args.max_len, args.limit)

    attn = "flash_attention_2"
    try:
        import flash_attn  # noqa
    except Exception:
        attn = "sdpa"

    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True)
        print("liger applied", flush=True)

    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation=attn)
    # text-only training: freeze vision stack
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") \
           or name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
            p.requires_grad = False
            n_frozen += p.numel()
    print("frozen params:", n_frozen / 1e6, "M", flush=True)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        logging_steps=10,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=2,
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        optim=args.optim,
        adam_beta2=0.95,
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    model.config.use_cache = True
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    # copy processor/vision configs so vLLM can load it
    import shutil
    for fn in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    print("saved", final)


if __name__ == "__main__":
    main()
