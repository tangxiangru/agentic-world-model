#!/usr/bin/env python3
"""GRPO RL on GSM8K *train* prompts with a verifiable correctness reward."""
import argparse
import json
import os
import random
import re

import torch
from datasets import Dataset

from common import SNAPSHOT, eval_fewshot_system, extract_pred, get_tokenizer, user_message


def build_dataset(path, tok, fewshot_prob, seed, limit=-1):
    items = [json.loads(l) for l in open(path)]
    if limit > 0:
        items = items[:limit]
    rng = random.Random(seed)
    from datasets import load_dataset
    devq = {json.loads(l)["question"] for l in open("data/dev.jsonl")} if os.path.exists("data/dev.jsonl") else set()
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    pool = []
    for rec in gsm:
        if rec["question"].strip() in devq:
            continue
        body, tgt = rec["answer"].split("####")
        pool.append(f"{rec['question'].strip()}\n\nReasoning:\n{body.strip()}\n\nANSWER: {tgt.strip()}")

    rows = []
    for it in items:
        msgs = []
        if rng.random() < fewshot_prob:
            k = rng.choice([2, 4, 6, 8, 10])
            msgs.append({"role": "system", "content": "\n\n".join(rng.sample(pool, k))})
        msgs.append({"role": "user", "content": user_message(it["question"])})
        rows.append({"prompt": msgs, "gold": it["answer"]})
    rng.shuffle(rows)
    return Dataset.from_list(rows)


def make_reward():
    def correctness(completions, gold, **kwargs):
        out = []
        for comp, g in zip(completions, gold):
            txt = comp[0]["content"] if isinstance(comp, list) else comp
            pred = extract_pred(txt)
            out.append(1.0 if (pred is not None and pred == g) else 0.0)
        return out
    correctness.__name__ = "correctness"

    def formatting(completions, **kwargs):
        out = []
        for comp in completions:
            txt = comp[0]["content"] if isinstance(comp, list) else comp
            out.append(0.15 if re.search(r"ANSWER:\s*-?[\d,]*\.?\d+\s*$", txt.strip()) else 0.0)
        return out
    formatting.__name__ = "formatting"

    return [correctness, formatting]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="ckpt/sft1")
    ap.add_argument("--out", default="ckpt/grpo")
    ap.add_argument("--data", default="data/gsm_train.jsonl")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--gen-bs", type=int, default=256)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--max-prompt", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--fewshot-prob", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--vllm-frac", type=float, default=0.22)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM

    tok = get_tokenizer()
    # gemma-3 chat turns end with <end_of_turn>, not <eos>; TRL needs this to
    # detect terminated completions (otherwise every completion looks truncated).
    tok.eos_token = "<end_of_turn>"
    print("eos:", tok.eos_token, tok.eos_token_id)
    ds = build_dataset(args.data, tok, args.fewshot_prob, args.seed, args.limit)
    print(ds)

    model = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad = False

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.num_gen,
        generation_batch_size=args.gen_bs,
        max_completion_length=args.max_completion,
        max_prompt_length=args.max_prompt,
        temperature=args.temp,
        top_p=1.0,
        beta=args.beta,
        loss_type="dapo",
        scale_rewards="group",
        mask_truncated_completions=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=0.5,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=6,
        report_to=[],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_frac,
        vllm_max_model_length=args.max_prompt + args.max_completion + 64,
        vllm_enable_sleep_mode=True,
        log_completions=True,
        num_completions_to_print=2,
        reward_weights=[1.0, 1.0],
        seed=args.seed,
        use_liger_kernel=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward(),
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    import shutil
    for fn in ["preprocessor_config.json", "processor_config.json", "generation_config.json"]:
        src = os.path.join(SNAPSHOT, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))
    print("saved", args.out)


if __name__ == "__main__":
    main()
