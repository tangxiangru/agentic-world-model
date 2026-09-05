"""Completion-only SFT for google/gemma-3-4b-pt on GSM8K-style data.

The training string is built with the grader's own chat template
(templates/gemma3.jinja) so training and grading render identically, and every
target ends with <end_of_turn> (token 106), which is what vLLM stops on
(generation_config.json eos_token_id = [1, 106]).
"""
import argparse
import hashlib
import json
import math
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
TEMPLATE_SHA = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

END_OF_TURN = "<end_of_turn>"


def build_prompt_str(user_text):
    # exactly what templates/gemma3.jinja renders for [user] + add_generation_prompt
    return f"<bos><start_of_turn>user\n{user_text.strip()}<end_of_turn>\n<start_of_turn>model\n"


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, fewshot_prefixes=None, fewshot_prob=0.0, seed=0):
        self.rows = []
        rng = random.Random(seed)
        n_trunc = 0
        lengths = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                user = d["prompt"]
                if fewshot_prefixes and rng.random() < fewshot_prob:
                    user = rng.choice(fewshot_prefixes) + "\n\n" + user
                p_ids = tok(build_prompt_str(user), add_special_tokens=False)["input_ids"]
                c_ids = tok(d["completion"], add_special_tokens=False)["input_ids"]
                assert c_ids[-1] == 106, f"target does not end with <end_of_turn>: {c_ids[-3:]}"
                c_ids = c_ids + [107]  # the '\n' the template emits after <end_of_turn>
                ids = p_ids + c_ids
                lengths.append(len(ids))
                if len(ids) > max_len:
                    n_trunc += 1
                    continue
                labels = [-100] * len(p_ids) + c_ids
                self.rows.append((ids, labels))
        lengths.sort()
        print(f"[data] {len(self.rows)} rows kept, {n_trunc} dropped for len>{max_len} "
              f"({n_trunc / max(1, len(lengths)):.3%}); p50={lengths[len(lengths)//2]} "
              f"p99={lengths[int(len(lengths)*0.99)]} max={lengths[-1]}", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * n)
            labels.append(f["labels"] + [-100] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parent", default=SNAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--master-fp32", action="store_true",
                    help="keep fp32 master weights and use bf16 autocast (better updates, more memory)")
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--fewshot-prob", type=float, default=0.0)
    args = ap.parse_args()

    assert hashlib.sha256(open(TEMPLATE_PATH, "rb").read()).hexdigest() == TEMPLATE_SHA, \
        "grading template changed - re-verify the rendering before training"

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(SNAP)

    fewshot_prefixes = None
    if args.fewshot_prob > 0:
        fewshot_prefixes = json.load(open("/home/ben/task/data/fewshot_prefixes.json"))
        print(f"[data] {len(fewshot_prefixes)} few-shot prefixes, p={args.fewshot_prob}")

    ds = SFTData(args.data, tok, args.max_seq_len, fewshot_prefixes, args.fewshot_prob, args.seed)

    dtype = torch.float32 if args.master_fp32 else torch.bfloat16
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=dtype, attn_implementation="eager"
    )
    model.config.use_cache = False
    # a greedy generation_config (do_sample=False + temperature/top_k set) is what we
    # ship for vLLM, but transformers refuses to SAVE it; keep a valid one while training
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid", do_sample=True, top_k=64, top_p=0.95,
    )
    # text-only training: freeze the vision stack so it neither trains nor drifts
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {type(model).__name__} trainable params {n_train/1e9:.2f}B dtype={dtype}")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy=("steps" if args.save_steps else "no"),
        save_steps=(args.save_steps or 500),
        save_total_limit=3,
        report_to=[],
        optim=args.optim,
        adam_beta2=0.98,
        max_grad_norm=1.0,
        seed=args.seed,
        dataloader_num_workers=4,
        group_by_length=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id)
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    # always ship bf16 weights: the grader loads final_model/ with vLLM at
    # gpu_memory_utilization 0.3, where an fp32 4B checkpoint does not fit
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    if hasattr(model.config, "text_config"):
        model.config.text_config.torch_dtype = "bfloat16"
    trainer.save_model(final)
    tok.save_pretrained(final)
    # the grader loads with vLLM from a fresh process: keep generation_config too
    import shutil
    for fn in ["generation_config.json", "preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    print("saved", final)


if __name__ == "__main__":
    main()
