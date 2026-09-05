"""GRPO (RLVR) on GSM8K *train* questions, starting from the SFT checkpoint.

Reward = 1 if the last number in the completion matches the gold answer (this is
exactly what inspect's match(location='end', numeric=True) scorer checks), with a
small bonus for the requested "ANSWER: x" final line.
"""
import argparse
import json
import os
import random

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from common import (
    BASE_SNAPSHOT,
    extract_answer,
    gsm8k_fewshots,
    load_chat_template,
    normalize_num,
    user_prompt,
)


def build_dataset(tok, n_limit, p_fewshot, seed=0, prompts_file=None):
    from datasets import load_dataset

    if prompts_file:
        src = [json.loads(l) for l in open(prompts_file)]
    else:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        src = [
            {"question": r["question"], "answer": r["answer"].split("####")[-1].strip()} for r in ds
        ]
    rng = random.Random(seed)
    system = "\n\n".join(gsm8k_fewshots(10, seed=42, shuffle=True))
    rows = []
    for r in src:
        a = normalize_num(r["answer"])
        if a is None:
            continue
        msgs = []
        if rng.random() < p_fewshot:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_prompt(r["question"].strip())})
        rows.append({"prompt": msgs, "answer": a})
    rng.shuffle(rows)
    if n_limit > 0:
        rows = rows[:n_limit]
    return Dataset.from_list(rows)


def make_reward(fmt_bonus):
    def reward_correct(completions, answer, **kwargs):
        out = []
        for c, gold in zip(completions, answer):
            text = c[0]["content"] if isinstance(c, list) else c
            r = 1.0 if extract_answer(text) == gold else 0.0
            if fmt_bonus:
                tail = text.strip().split("\n")[-1].strip()
                if tail.startswith("ANSWER:"):
                    r += fmt_bonus
            out.append(r)
        return out

    reward_correct.__name__ = "correct"
    return reward_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default="runs/grpo_v1")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--p-fewshot", type=float, default=0.15)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion", type=int, default=640)
    ap.add_argument("--max-prompt", type=int, default=2600)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--vllm-util", type=float, default=0.25)
    ap.add_argument("--fmt-bonus", type=float, default=0.05)
    ap.add_argument("--optim", default="adamw_bnb_8bit")
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--sleep-mode", type=int, default=1)
    ap.add_argument("--steps-per-gen", type=int, default=1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT)
    tok.chat_template = load_chat_template()
    tok.padding_side = "left"
    # the SFT model terminates turns with <end_of_turn> (106), not <eos> (1);
    # TRL uses processing_class.eos_token_id to detect non-truncated completions.
    tok.eos_token = "<end_of_turn>"

    ds = build_dataset(tok, args.limit, args.p_fewshot, prompts_file=args.prompts_file)
    print("prompts:", len(ds))

    from liger_kernel.transformers import apply_liger_kernel_to_gemma3

    apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=False)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        adam_beta2=0.95,
        weight_decay=0.0,
        max_grad_norm=0.5,
        bf16=True,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        generation_batch_size=args.gen_batch,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        top_p=1.0,
        beta=args.beta,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="dr_grpo",
        scale_rewards="none",
        mask_truncated_completions=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_util,
        vllm_max_model_length=4096,
        vllm_enable_sleep_mode=bool(args.sleep_mode),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        log_completions=False,
    )

    trainer = GRPOTrainer(
        model=args.init,
        reward_funcs=[make_reward(args.fmt_bonus)],
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    import shutil

    # ship the pristine tokenizer (we retargeted tok.eos_token only for TRL's
    # truncation bookkeeping) plus the processor configs vLLM expects
    for fn in (
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        src = os.path.join(BASE_SNAPSHOT, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(final, fn))
    print("saved", final)


if __name__ == "__main__":
    main()
