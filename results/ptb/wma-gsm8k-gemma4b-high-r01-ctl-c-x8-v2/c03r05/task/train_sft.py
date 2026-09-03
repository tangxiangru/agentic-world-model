#!/usr/bin/env python3
"""Completion-only SFT for google/gemma-3-4b-pt on GSM8K-style data.

Prompts are rendered with the *grader's own* chat template
(templates/gemma3.jinja, read from disk and md5-printed) so training and
grading see byte-identical strings -- the `template_unreachable` pitfall.

Every target ends with <end_of_turn>. The saved checkpoint's tokenizer eos is
rewritten to <end_of_turn> (id 106) so the vLLM server the grader starts stops
on exactly the token the model was trained to emit, and generation_config.json
is rewritten to greedy so the grader does not sample at T=1.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, Gemma3ForConditionalGeneration,
                          Trainer, TrainingArguments)

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
TEMPLATE_PATH = "templates/gemma3.jinja"

# byte-for-byte from inspect_evals/gsm8k/gsm8k.py MATH_PROMPT_TEMPLATE
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_OF_TURN = "<end_of_turn>"
EOT_ID = 106


def render(tok, question: str, system: str | None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user",
                 "content": MATH_PROMPT_TEMPLATE.format(prompt=question)})
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


class Packed(Dataset):
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.labels[i]}


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


def save_for_grader(model, tok, path):
    """Save weights + a tokenizer/generation_config the grader's vLLM honours."""
    os.makedirs(path, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(path, safe_serialization=True)
    tok.eos_token = END_OF_TURN            # so vLLM stops on what we trained
    tok.save_pretrained(path)
    # keep the base tokenizer.json byte-for-byte: only tokenizer_config.json
    # should differ from the immutable snapshot
    shutil.copy2(os.path.join(BASE, "tokenizer.json"),
                 os.path.join(path, "tokenizer.json"))
    gen = {"bos_token_id": 2, "eos_token_id": [1, EOT_ID], "pad_token_id": 0,
           "do_sample": False, "temperature": 0.0, "cache_implementation": "hybrid"}
    with open(os.path.join(path, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    # keep the processor files so the multimodal config still loads
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, fn)
        dst = os.path.join(path, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            with open(src) as a, open(dst, "w") as b:
                b.write(a.read())
    print("saved", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--parent", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2816)
    ap.add_argument("--fewshot-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--liger", type=int, default=1)
    ap.add_argument("--gc", type=int, default=1)
    args = ap.parse_args()

    tpl = open(TEMPLATE_PATH).read()
    print("template md5:", hashlib.md5(tpl.encode()).hexdigest())

    tok = AutoTokenizer.from_pretrained(args.parent)
    tok.chat_template = tpl
    assert tok.convert_tokens_to_ids(END_OF_TURN) == EOT_ID

    grader_fewshot = open("data/fewshot_system.txt").read()
    pool = [json.loads(l) for l in open("data/fewshot_pool.jsonl")]

    rng = random.Random(args.seed)
    rows = []
    with open(args.data) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    prompts, completions = [], []
    for r in rows:
        sysmsg = None
        if rng.random() < args.fewshot_frac:
            if rng.random() < 0.3:
                sysmsg = grader_fewshot
            else:
                k = rng.randint(1, 8)
                sysmsg = "\n\n".join(x["shot"] for x in rng.sample(pool, k))
        prompts.append(render(tok, r["question"], sysmsg))
        assert r["target"].endswith(END_OF_TURN)
        completions.append(r["target"])

    print(f"{len(rows)} training rows; tokenizing...")
    p_enc = tok(prompts, add_special_tokens=False)["input_ids"]
    c_enc = tok(completions, add_special_tokens=False)["input_ids"]
    ids, labels, n_trunc = [], [], 0
    for p, c in zip(p_enc, c_enc):
        i = p + c
        l = [-100] * len(p) + c[:]
        if len(i) > args.max_len:
            n_trunc += 1
            i, l = i[-args.max_len:], l[-args.max_len:]
        ids.append(i)
        labels.append(l)
    lens = sorted(len(i) for i in ids)
    print(f"len p50={lens[len(lens)//2]} p99={lens[int(.99*len(lens))]} "
          f"max={lens[-1]} truncated={n_trunc} ({n_trunc/len(ids):.3%}) "
          f"total_tokens={sum(lens)/1e6:.1f}M")
    assert n_trunc / len(ids) < 0.02, "too many truncated rows"
    assert all(l[-1] == EOT_ID for l in labels), "a target does not end in <end_of_turn>"

    print("---- example prompt ----")
    print(prompts[0][:900])
    print("---- example completion tail ----")
    print(repr(completions[0][-200:]))

    if args.liger:
        # gemma-3's vocab is 262k: materialising fp32 logits for a whole
        # micro-batch OOMs an 80GB H100 at any useful batch size. Liger's fused
        # linear cross-entropy never materialises them.
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=True,
                                     cross_entropy=False)
        print("liger fused-linear-CE enabled")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.parent, dtype=torch.bfloat16, attn_implementation=args.attn)
    model.config.use_cache = False
    # A parent that we ourselves saved carries the greedy generation_config
    # (do_sample=False + temperature=0.0). HF's GenerationConfig.save_pretrained
    # rejects that combination, so every checkpoint save would raise. Replace it
    # with a valid one here; save_for_grader writes the greedy file at the end.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, EOT_ID], pad_token_id=0, do_sample=False)
    n_frozen = 0
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad_(False)
            n_frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"frozen {n_frozen/1e6:.0f}M, trainable {trainable/1e6:.0f}M")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=bool(args.gc),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        group_by_length=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=Packed(ids, labels),
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()
    save_for_grader(model, tok, os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
