#!/usr/bin/env python3
"""Completion-only SFT of google/gemma-3-4b-pt for GSM8K.

Prompts are rendered with the *exact* chat template the grader uses
(templates/gemma3.jinja, sha256-checked) and the exact prompt wording from
inspect_evals.gsm8k, so training and grading render the same strings.
Targets end with <end_of_turn> (token 106), which is in the base model's
generation_config.eos_token_id, i.e. the token vLLM stops on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE_PATH = "/home/ben/task/templates/gemma3.jinja"
TEMPLATE_SHA256 = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"

# verbatim from inspect_evals/gsm8k/gsm8k.py
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"


def load_template() -> str:
    with open(TEMPLATE_PATH, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    print(f"[template] {TEMPLATE_PATH} sha256={digest}")
    if TEMPLATE_SHA256 and digest != TEMPLATE_SHA256:
        raise SystemExit(
            f"chat template changed: expected {TEMPLATE_SHA256}, got {digest}"
        )
    return raw.decode()


def gsm8k_fewshot_pool():
    """Few-shot exemplars in the exact shape inspect's sample_to_fewshot() builds."""
    import pyarrow.parquet as pq
    import glob

    f = glob.glob(
        "/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-00000-of-00001.parquet"
    )[0]
    d = pq.read_table(f).to_pydict()
    out = []
    for q, a in zip(d["question"], d["answer"]):
        parts = a.split("####")
        target = parts[-1].strip()
        reasoning = "####".join(parts[:-1]).strip()
        out.append(f"{q}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--n-rows", type=int, default=60000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dry-run", action="store_true", help="build data, print stats, exit")
    args = ap.parse_args()

    template = load_template()
    tok = AutoTokenizer.from_pretrained(BASE)
    eot_id = tok.convert_tokens_to_ids(END_OF_TURN)
    print(f"[tok] {END_OF_TURN} -> {eot_id}; eos={tok.eos_token}({tok.eos_token_id})")
    assert eot_id == 106, eot_id

    rows = [json.loads(l) for l in open(args.data)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_rows]
    pool = gsm8k_fewshot_pool()

    feats = {"input_ids": [], "labels": [], "length": []}
    n_trunc = 0
    n_fewshot = 0
    sample_render = None
    for i, r in enumerate(rows):
        user = MATH_PROMPT_TEMPLATE.format(prompt=r["problem"])
        msgs = []
        if rng.random() < args.fewshot_frac:
            k = rng.randint(2, 10)
            shots = rng.sample(pool, k)
            msgs.append({"role": "system", "content": "\n\n".join(shots)})
            n_fewshot += 1
        msgs.append({"role": "user", "content": user})
        prompt = tok.apply_chat_template(
            msgs, chat_template=template, tokenize=False, add_generation_prompt=True
        )
        target = r["completion"]
        assert target.endswith(END_OF_TURN), target[-40:]
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        if sample_render is None:
            sample_render = (prompt, target, p_ids[:4], t_ids[-4:])
        if len(p_ids) + len(t_ids) > args.max_seq_len:
            n_trunc += 1
            continue
        ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        feats["input_ids"].append(ids)
        feats["labels"].append(labels)
        feats["length"].append(len(ids))

    print(f"[data] kept {len(feats['input_ids'])} / {len(rows)}; dropped-too-long {n_trunc} "
          f"({n_trunc/max(1,len(rows)):.3%}); fewshot rows {n_fewshot}")
    ls = sorted(feats["length"])
    print(f"[data] tokens p50={ls[len(ls)//2]} p90={ls[int(.9*len(ls))]} max={ls[-1]} "
          f"total={sum(ls)/1e6:.1f}M")
    print("[render] PROMPT >>>\n" + sample_render[0][:1200])
    print("[render] <<< TARGET >>>\n" + sample_render[1][-400:])
    print(f"[render] first prompt ids {sample_render[2]} last target ids {sample_render[3]} "
          f"(last must be {eot_id})")
    assert feats["labels"][0][-1] == eot_id
    if args.dry_run:
        return

    ds = Dataset.from_dict(feats)

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append([1] * len(b["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

    model = AutoModelForCausalLM.from_pretrained(
        args.parent, dtype=torch.float32, attn_implementation=args.attn
    )
    frozen = 0
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {type(model).__name__} trainable={trainable/1e9:.2f}B frozen={frozen/1e6:.0f}M")
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        use_liger_kernel=not args.no_liger,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        length_column_name="length",
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=4,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()

    final = os.path.join(args.out, "final")
    os.makedirs(final, exist_ok=True)
    model.config.use_cache = True
    model = model.to(torch.bfloat16)
    model.save_pretrained(final, safe_serialization=True)
    for fn in os.listdir(BASE):
        if fn.endswith(".safetensors") or fn.endswith(".index.json"):
            continue
        if fn in ("config.json", "generation_config.json", "README.md"):
            continue
        shutil.copy(os.path.join(BASE, fn), os.path.join(final, fn))
    print(f"[save] {final}")
    print(json.dumps(sorted(os.listdir(final))))


if __name__ == "__main__":
    main()
