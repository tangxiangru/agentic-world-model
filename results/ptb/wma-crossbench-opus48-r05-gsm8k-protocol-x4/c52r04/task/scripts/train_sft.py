#!/usr/bin/env python3
"""Full SFT of google/gemma-3-4b-pt on GSM8K-train-style data.

Completion-only loss. Prompt is rendered with the SAME templates/gemma3.jinja the
grader uses; the target ends with <end_of_turn> (id 106, an eos id in the model's
generation_config) so vLLM stops cleanly at grading time.
"""
import argparse, json, os
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Gemma3ForConditionalGeneration, Trainer, TrainingArguments)
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, template, max_seq_len):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tokenizer
        self.template = template
        self.max_seq_len = max_seq_len
        self.eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        self.examples = []
        n_trunc = 0
        for r in self.rows:
            pid = tokenizer.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                chat_template=template, add_generation_prompt=True, tokenize=True)
            comp = r["completion"].strip()
            # The completion already ends with the literal <end_of_turn> (id 106).
            cid = tokenizer(comp, add_special_tokens=False).input_ids
            assert cid[-1] == self.eot, f"completion must end with <end_of_turn>: {cid[-3:]}"
            input_ids = pid + cid
            labels = [-100] * len(pid) + list(cid)
            if len(input_ids) > max_seq_len:
                n_trunc += 1
                continue  # drop; measured to be 0% at 1024
            self.examples.append({"input_ids": input_ids, "labels": labels})
        print(f"[data] kept {len(self.examples)} / {len(self.rows)} rows "
              f"(dropped {n_trunc} over max_seq_len={max_seq_len})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


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
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_strategy", default="epoch")
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    template = open("templates/gemma3.jinja").read()

    ds = SFTDataset(args.data, tok, template, args.max_seq_len)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager")
    # Freeze vision tower + multimodal projector; train language model only.
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        if ("vision" in name) or ("multi_modal" in name) or ("mm_" in name):
            p.requires_grad = False
            n_freeze += p.numel()
        else:
            n_train += p.numel()
    print(f"[model] trainable params: {n_train/1e9:.3f}B  frozen: {n_freeze/1e9:.3f}B")
    model.config.use_cache = False
    # Sanitize generation_config: a parent checkpoint saved with a greedy eval config
    # (do_sample=False, temperature=0.0) fails GenerationConfig validation at save time.
    # Reset to a valid sampling default (mirrors the base model); eval overrides to greedy later.
    gc = model.generation_config
    gc.do_sample = True
    gc.temperature = 1.0
    gc.top_p = 0.95
    gc.top_k = 64
    print(f"[model] sanitized generation_config: do_sample={gc.do_sample} temperature={gc.temperature}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        seed=args.seed,
        optim="adamw_torch",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    _copy_processor_configs(args.model, args.out)
    print("[done] saved to", args.out)


def _copy_processor_configs(src, dst):
    """vLLM loads this multimodal Gemma3 checkpoint and needs the image-processor
    configs; Trainer.save_model does not write them. Copy from the base model."""
    import shutil, glob
    dsts = [dst] + glob.glob(os.path.join(dst, "checkpoint-*"))
    for fn in ("preprocessor_config.json", "processor_config.json"):
        s = os.path.join(src, fn)
        if not os.path.exists(s):
            continue
        for d in dsts:
            if os.path.isdir(d):
                shutil.copy(s, os.path.join(d, fn))
        print(f"[save] copied {fn} to {len(dsts)} dir(s)")


if __name__ == "__main__":
    main()
