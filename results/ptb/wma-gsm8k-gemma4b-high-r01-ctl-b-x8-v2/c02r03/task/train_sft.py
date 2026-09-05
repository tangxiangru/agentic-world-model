#!/usr/bin/env python3
"""Full fine-tune of google/gemma-3-4b-pt on GSM8K-style CoT.

Renders every training row with the *exact* chat template the grader uses
(templates/gemma3.jinja, hash-checked) so training and grading agree
byte-for-byte, masks the loss to the completion, and ends every target with
<end_of_turn> -- the token vLLM stops on (generation_config eos_token_id
contains 106).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"
TEMPLATE_SHA256 = None  # filled at first run, printed for the card

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"


def load_template() -> tuple[str, str]:
    raw = open(TEMPLATE, "rb").read()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


class SFTDataset(Dataset):
    def __init__(self, rows, tok, max_seq_len):
        self.tok = tok
        self.max_seq_len = max_seq_len
        self.examples = []
        n_trunc = 0
        eot_id = tok.convert_tokens_to_ids(END_OF_TURN)
        for r in rows:
            msgs = []
            if r.get("system"):
                msgs.append({"role": "system", "content": r["system"]})
            msgs.append({"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])})
            prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            t_ids = tok(r["target"], add_special_tokens=False)["input_ids"] + [eot_id]
            ids = p_ids + t_ids
            if len(ids) > max_seq_len:
                n_trunc += 1
                continue
            labels = [-100] * len(p_ids) + list(t_ids)
            self.examples.append({"input_ids": ids, "labels": labels, "length": len(ids)})
        self.n_dropped = n_trunc

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class TokenBudgetBatches:
    """Micro-batches of variable size, each holding at most `budget` padded tokens.

    Length-bucketed so padding waste stays small, then the batch order is
    shuffled so the optimizer does not see all short rows first. This is what
    keeps the 262k-vocab logits tensor bounded while still filling the GPU on
    the (mostly ~330 token) short rows.
    """

    def __init__(self, lengths, budget, seed, drop_last=False):
        self.batches = []
        rng = random.Random(seed)
        idx = list(range(len(lengths)))
        rng.shuffle(idx)
        chunk = 8192
        for c0 in range(0, len(idx), chunk):
            part = sorted(idx[c0 : c0 + chunk], key=lambda i: lengths[i])
            cur, cur_max = [], 0
            for i in part:
                m = max(cur_max, lengths[i])
                if cur and m * (len(cur) + 1) > budget:
                    self.batches.append(cur)
                    cur, cur_max = [i], lengths[i]
                else:
                    cur, cur_max = cur + [i], m
            if cur:
                self.batches.append(cur)
        rng.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            k = m - len(f["input_ids"])
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
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--token-budget", type=int, default=6144)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tmpl, sha = load_template()
    print("chat template sha256:", sha, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = tmpl

    rows = [json.loads(l) for l in open(args.data)]
    random.Random(args.seed).shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]

    ds = SFTDataset(rows, tok, args.max_seq_len)
    lens = sorted(e["length"] for e in ds.examples)
    print(
        f"rows={len(ds)} dropped_over_maxlen={ds.n_dropped} "
        f"({ds.n_dropped / max(1, len(rows)):.4%}) tok_p50={lens[len(lens)//2]} "
        f"tok_p99={lens[int(len(lens)*0.99)]} tok_max={lens[-1]} total_tokens={sum(lens)}",
        flush=True,
    )

    # show one rendered example end-to-end (template_unreachable pitfall)
    ex = ds.examples[0]
    print("---- rendered example (tail) ----")
    print(repr(tok.decode(ex["input_ids"][-200:])))
    print("---- loss starts at token", ex["labels"].index(next(x for x in ex["labels"] if x != -100)), "----", flush=True)

    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa"
    )
    print("model class:", type(model).__name__, flush=True)
    model.config.use_cache = False
    # freeze the vision tower / projector: text-only training, saves grads+optimizer state
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen params {n_frozen/1e6:.1f}M  trainable {n_train/1e9:.2f}B", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    collator = Collator(tok.pad_token_id)
    batches = TokenBudgetBatches([e["length"] for e in ds.examples], args.token_budget, args.seed)
    print(f"micro-batches/epoch={len(batches)} median_bs={sorted(len(b) for b in batches)[len(batches)//2]}", flush=True)

    class BudgetTrainer(Trainer):
        def get_train_dataloader(self):
            dl = DataLoader(
                ds,
                batch_sampler=batches,
                collate_fn=collator,
                num_workers=2,
                pin_memory=True,
            )
            return self.accelerator.prepare(dl)

    trainer = BudgetTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    # transformers validates generation_config on save and rejects do_sample=False with
    # temperature/top_k set -- which is exactly the greedy config we write below. Reset it
    # to a minimal valid one first, then write the file we actually want afterwards.
    model.generation_config = GenerationConfig(
        bos_token_id=tok.bos_token_id, eos_token_id=[1, 106], pad_token_id=tok.pad_token_id
    )
    model.to(torch.bfloat16)  # match the base checkpoint dtype so vLLM loads bf16
    model.config.torch_dtype = "bfloat16"
    model.config.dtype = "bfloat16"
    trainer.save_model(final)
    tok.save_pretrained(final)
    # greedy decoding: vLLM reads generation_config.json for default sampling params
    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    gc.pop("cache_implementation", None)
    json.dump(gc, open(os.path.join(final, "generation_config.json"), "w"), indent=2)
    # the grader loads the model directory directly; keep the preprocessor files too
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            json.dump(json.load(open(src)), open(os.path.join(final, f), "w"), indent=2)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
