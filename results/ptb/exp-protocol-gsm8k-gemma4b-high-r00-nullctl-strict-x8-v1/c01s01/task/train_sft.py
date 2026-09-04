#!/usr/bin/env python3
import argparse, json, os, random, math
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig,
                          Trainer, TrainingArguments)

BASE = os.path.expanduser(
    "~/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, limit=None):
        self.rows = []
        skipped = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                # gemma3 chat template rendering, done manually for speed/control
                p = "<bos><start_of_turn>user\n" + r["prompt"].strip() + "<end_of_turn>\n<start_of_turn>model\n"
                c = r["completion"].strip() + "<end_of_turn>\n"
                pi = tok(p, add_special_tokens=False)["input_ids"]
                ci = tok(c, add_special_tokens=False)["input_ids"]
                if len(pi) + len(ci) > max_len:
                    skipped += 1
                    continue
                self.rows.append((pi, ci))
        print(f"loaded {len(self.rows)} rows (skipped {skipped} too long)")
        self.lengths = [len(a) + len(b) for a, b in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pi, ci = self.rows[i]
        ids = pi + ci
        labels = [-100] * len(pi) + ci[:]
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        m = ((m + 7) // 8) * 8
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            pad = m - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1792)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-gc", action="store_true")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = SFTData(args.data, tok, args.max_len, args.limit)

    model = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=(torch.float32 if args.dtype=="fp32" else torch.bfloat16),
        attn_implementation=args.attn)
    print(type(model))
    # freeze vision tower / projector: text-only training
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"frozen params: {n_frozen/1e6:.1f}M ; trainable: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        gradient_checkpointing=not args.no_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        report_to=[],
        use_liger_kernel=not args.no_liger,
        seed=17,
        max_steps=args.max_steps,
    )

    class T(Trainer):
        def _get_train_sampler(self, *a, **k):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=ds.lengths, generator=torch.Generator().manual_seed(17))

    trainer = T(model=model, args=targs, train_dataset=ds,
                data_collator=Collator(tok.pad_token_id))
    trainer.train()
    print("max mem GB:", torch.cuda.max_memory_allocated()/1e9)
    model.config.use_cache = True
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
