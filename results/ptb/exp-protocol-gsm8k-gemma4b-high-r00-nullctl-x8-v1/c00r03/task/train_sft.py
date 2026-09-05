#!/usr/bin/env python3
"""Full fine-tune google/gemma-3-4b-pt for GSM8K-style math CoT."""
import argparse, json, os, random, shutil

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig,
                          Trainer, TrainingArguments)

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

END_TURN = "<end_of_turn>"


def build_fewshot_pool():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        ans = r["answer"].split("####")
        target = ans.pop().strip()
        reasoning = "####".join(ans).strip()
        out.append(f"{r['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return out


class SFTData(Dataset):
    def __init__(self, path, tok, max_len, fewshot_frac, fewshot_pool, seed=0,
                 limit=None):
        self.tok = tok
        self.max_len = max_len
        rows = [json.loads(l) for l in open(path)]
        if limit:
            rows = rows[:limit]
        rng = random.Random(seed)
        self.ex = []
        skipped = 0
        for r in rows:
            sys_msg = None
            if fewshot_pool and rng.random() < fewshot_frac:
                k = rng.randint(2, 10)
                sys_msg = "\n\n".join(rng.sample(fewshot_pool, k))
            msgs = []
            if sys_msg:
                msgs.append({"role": "system", "content": sys_msg})
            msgs.append({"role": "user",
                         "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"])})
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
            comp = r["solution"].strip() + END_TURN
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(comp, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(c_ids) > max_len:
                skipped += 1
                continue
            self.ex.append((p_ids, c_ids))
        print(f"dataset: {len(self.ex)} examples ({skipped} too long)", flush=True)
        self.lengths = [len(a) + len(b) for a, b in self.ex]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        p, c = self.ex[i]
        ids = p + c
        labels = [-100] * len(p) + list(c)
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        n = ((n + 15) // 16) * 16
        ids, lab, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            lab.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--out", default="ckpt/gemma_sft_v1")
    ap.add_argument("--init", default=BASE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--fewshot-frac", type=float, default=0.12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("templates/gemma3.jinja").read()

    pool = build_fewshot_pool() if args.fewshot_frac > 0 else None
    ds = SFTData(args.data, tok, args.max_len, args.fewshot_frac, pool,
                 seed=args.seed, limit=args.limit)

    from liger_kernel.transformers import (apply_liger_kernel_to_gemma3,
                                           apply_liger_kernel_to_gemma3_text)
    apply_liger_kernel_to_gemma3()
    apply_liger_kernel_to_gemma3_text()

    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.float32, attn_implementation="sdpa")
    # freeze the (unused) vision stack
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    model = model.to(torch.bfloat16)
    # a greedy generation_config (from a patched --init dir) fails HF validation
    # on save; write a plain sampling one here and re-apply greedy afterwards.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid", do_sample=True, top_k=64, top_p=0.95)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    for f in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
