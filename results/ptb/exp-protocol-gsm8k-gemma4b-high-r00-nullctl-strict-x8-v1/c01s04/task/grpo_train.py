#!/usr/bin/env python3
"""GRPO on GSM8K-train-style problems with an exact-match numeric reward."""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from datasets import Dataset
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(s):
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else ("%f" % f).rstrip("0").rstrip(".")
    except Exception:
        return s


def extract(text):
    m = re.findall(r"ANSWER:\s*([^\n]*)", text)
    if m:
        nums = NUM_RE.findall(m[-1])
        if nums:
            return norm(nums[-1])
    return None


def reward_correct(completions, answer, **kw):
    out = []
    for c, a in zip(completions, answer):
        txt = c[0]["content"] if isinstance(c, list) else c
        out.append(1.0 if extract(txt) == norm(a) else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", default="data/rft_questions.jsonl")
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--vllm-util", type=float, default=0.28)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--save-steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.questions)][: args.limit]
    ds = Dataset.from_list([
        {"prompt": [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=r["question"].strip())}],
         "answer": norm(r["answer"])}
        for r in rows
    ])
    print(ds)

    tok = AutoTokenizer.from_pretrained(args.init)
    tok.chat_template = open("templates/gemma3.jinja").read()
    # the model terminates turns with <end_of_turn> (106); make TRL treat that as EOS
    tok.eos_token = "<end_of_turn>"
    print("eos id for trl:", tok.eos_token_id)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.init, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup,
        adam_beta2=0.99,
        max_grad_norm=0.5,
        weight_decay=0.0,
        bf16=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.num_gen,
        generation_batch_size=args.gen_batch,
        max_prompt_length=512,
        max_completion_length=args.max_completion,
        temperature=args.temp,
        top_p=0.95,
        top_k=64,
        beta=args.beta,
        loss_type="dr_grpo",
        scale_rewards=False,
        mask_truncated_completions=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_util,
        vllm_max_model_length=1536,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=12,
        report_to=[],
        log_completions=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[reward_correct],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    import shutil
    for f in ["preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print("saved", args.out)


if __name__ == "__main__":
    main()
