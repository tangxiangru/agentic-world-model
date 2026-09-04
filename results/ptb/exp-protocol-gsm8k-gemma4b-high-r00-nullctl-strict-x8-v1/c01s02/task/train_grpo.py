#!/usr/bin/env python3
"""GRPO (verifiable-reward RL) on GSM8K train questions, starting from the SFT model."""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

PROMPT = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text):
    m = NUM_RE.findall(text or "")
    if not m:
        return None
    try:
        return round(float(m[-1].replace(",", "").rstrip(".")), 4)
    except ValueError:
        return None


def correctness(completions, answer, **kwargs):
    out = []
    for c, a in zip(completions, answer):
        txt = c if isinstance(c, str) else c[0]["content"]
        gold = last_number(a)
        pred = last_number(txt)
        r = 1.0 if (gold is not None and pred == gold) else 0.0
        if "ANSWER:" not in txt:
            r -= 0.2
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="runs/grpo1")
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--vllm-frac", type=float, default=0.28)
    ap.add_argument("--filter-easy", type=int, default=0,
                    help="drop problems solved this many times or more out of 16 in RFT stats")
    ap.add_argument("--stats", default="data/rft1_stats.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()

    tr = load_dataset("openai/gsm8k", "main", split="train")
    keep = None
    if args.filter_easy and os.path.exists(args.stats):
        pr = {}
        for l in open(args.stats):
            d = json.loads(l)
            pr[d["question"]] = d["correct"]
        keep = pr

    rows = []
    for r in tr:
        q = r["question"].strip()
        if keep is not None and keep.get(q, 0) >= args.filter_easy:
            continue
        rows.append({
            "prompt": [{"role": "user", "content": PROMPT.format(prompt=q)}],
            "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
        })
    ds = Dataset.from_list(rows).shuffle(seed=0)
    print("GRPO prompts:", len(ds))

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.num_gen,
        max_completion_length=args.max_completion,
        max_prompt_length=512,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        beta=args.beta,
        temperature=args.temp,
        top_p=1.0,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_frac,
        report_to=[],
        seed=0,
        log_completions=False,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=correctness,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
