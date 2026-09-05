#!/usr/bin/env python3
"""GRPO (LoRA) on GSM8K-train-style problems, rewarding a correct final ANSWER."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from prep_data import MATH_PROMPT_TEMPLATE
from train_sft import BASE

ANS_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.M)


def norm_num(s: str):
    s = s.strip().strip("$").replace(",", "").replace("%", "").rstrip(".").replace(" ", "")
    try:
        return round(float(s), 6)
    except ValueError:
        return None


def reward_correct(completions, answer, **kwargs):
    out = []
    for c, gold in zip(completions, answer):
        g = norm_num(gold)
        ms = ANS_RE.findall(c)
        if not ms:
            out.append(-0.5)
            continue
        p = norm_num(ms[-1])
        out.append(1.0 if (p is not None and g is not None and p == g) else 0.0)
    return out


def build_dataset(n_aug: int, seed: int, keep_gsm8k: bool = True, prompt_file: str | None = None):
    from datasets import load_dataset

    rows = []
    if prompt_file:
        with open(prompt_file) as f:
            for line in f:
                r = json.loads(line)
                rows.append((r["problem"].strip(), r["answer"].strip()))
        random.Random(seed + 1).shuffle(rows)
        return _to_ds(rows)
    if keep_gsm8k:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for rec in ds:
            rows.append((rec["question"].strip(), rec["answer"].split("####")[-1].strip()))
    if n_aug > 0:
        import pyarrow.parquet as pq

        files = sorted(
            glob.glob(
                "/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/"
                "snapshots/*/data/train_1M-*.parquet"
            )
        )
        pool, seen = [], set()
        for f in files:
            df = pq.read_table(f).to_pandas()
            sub = df[df.problem_source == "augmented_gsm8k"]
            for p, a in zip(sub["problem"], sub["expected_answer"]):
                k = p.strip().lower()
                if k in seen or len(p) > 1000:
                    continue
                seen.add(k)
                pool.append((p.strip(), a.strip()))
        random.Random(seed).shuffle(pool)
        rows.extend(pool[:n_aug])
    random.Random(seed + 1).shuffle(rows)
    return _to_ds(rows)


def _to_ds(rows):
    return Dataset.from_dict(
        {
            "prompt": [
                "<start_of_turn>user\n"
                + MATH_PROMPT_TEMPLATE.format(prompt=q)
                + "<end_of_turn>\n<start_of_turn>model\n"
                for q, _ in rows
            ],
            "answer": [a for _, a in rows],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default="ckpt/grpo_v1")
    ap.add_argument("--n-aug", type=int, default=8000)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-completion", type=int, default=400)
    ap.add_argument("--max-prompt", type=int, default=320)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--vllm-frac", type=float, default=0.30)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--prompt-file", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    ds = build_dataset(args.n_aug, args.seed, prompt_file=args.prompt_file)
    print(ds)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        exclude_modules=r".*vision_tower.*|.*multi_modal_projector.*",
    )

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        weight_decay=0.0,
        max_grad_norm=0.2,
        bf16=True,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        temperature=args.temp,
        top_p=0.95,
        top_k=64,
        beta=args.beta,
        loss_type="dr_grpo",
        scale_rewards="none",
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_frac,
        vllm_max_model_length=args.max_prompt + args.max_completion + 8,
        generation_kwargs={"stop_token_ids": [106, 1]},
        report_to=[],
        seed=args.seed,
        log_completions=True,
        num_completions_to_print=1,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[reward_correct],
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.out)
    print("saved adapter to", args.out)


if __name__ == "__main__":
    main()
