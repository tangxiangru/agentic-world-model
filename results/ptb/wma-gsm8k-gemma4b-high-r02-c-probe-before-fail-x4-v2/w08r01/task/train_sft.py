"""Full-parameter SFT of google/gemma-3-4b-pt on GSM8K-style data.

Rows are rendered with templates/gemma3.jinja -- the *same* file evaluate.py
hands to vLLM -- so the training string and the graded string cannot drift.
Loss is computed on the assistant turn only (prompt tokens are masked to -100);
every target is terminated by <end_of_turn> (token 106), which is in the base
checkpoint's generation_config.eos_token_id, so vLLM stops there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "templates/gemma3.jinja"
END_OF_TURN = "<end_of_turn>"


class SFTRows(Dataset):
    def __init__(self, path: str, tok, template: str, max_len: int):
        cache = f"{path}.tok{max_len}.pt"
        if os.path.exists(cache):
            blob = torch.load(cache)
            self.rows, self.n_trunc = blob["rows"], blob["n_trunc"]
            print(f"{len(self.rows)} rows loaded from cache {cache}; {self.n_trunc} previously dropped")
            return
        self.rows = []
        n_trunc = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                msgs = r["messages"]
                prompt = tok.apply_chat_template(
                    msgs[:-1], chat_template=template, tokenize=False, add_generation_prompt=True
                )
                full = tok.apply_chat_template(msgs, chat_template=template, tokenize=False)
                assert full.startswith(prompt)
                # cut the trailing newline after <end_of_turn>: nothing is generated past eos
                full = full[: full.rindex(END_OF_TURN) + len(END_OF_TURN)]
                p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                f_ids = tok(full, add_special_tokens=False)["input_ids"]
                if len(f_ids) > max_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
                self.rows.append({"input_ids": f_ids, "labels": labels})
        self.n_trunc = n_trunc
        torch.save({"rows": self.rows, "n_trunc": n_trunc}, cache)
        print(f"{len(self.rows)} rows loaded from {path}; {n_trunc} dropped for exceeding max_len={max_len}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            attn.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--out", default="ckpts/exp-02")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: keep only N rows")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--save-steps", type=int, default=0, help="0 = save once per epoch")
    args = ap.parse_args()

    template = open(TEMPLATE).read()
    print("template sha256:", hashlib.sha256(template.encode()).hexdigest())

    tok = AutoTokenizer.from_pretrained(SNAP)
    ds = SFTRows(args.data, tok, template, args.max_len)
    if args.limit:
        ds.rows = ds.rows[: args.limit]

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False
    # text-only training: freeze the vision stack so it keeps no optimizer state
    for name, p in model.named_parameters():
        if name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e9:.2f}B of {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

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
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps or 500,
        save_total_limit=None,
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
        optim="adamw_torch_fused",
        use_liger_kernel=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(tok.pad_token_id),
        processing_class=tok,
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    # transformers "aligns" generation_config to the tokenizer on load, which drops
    # <end_of_turn> (106) from eos_token_id. Put the base checkpoint's decode config
    # back before saving, or vLLM will never stop at the token we trained on.
    base_gen = GenerationConfig.from_pretrained(SNAP)
    trainer.model.generation_config = base_gen
    trainer.save_model(final)
    tok.save_pretrained(final)
    try:
        AutoProcessor.from_pretrained(SNAP).save_pretrained(final)
    except Exception as e:  # processor is only needed for images, but vLLM loads it
        print("processor save skipped:", e)
    print("saved", final)


if __name__ == "__main__":
    main()
