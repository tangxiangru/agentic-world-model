#!/usr/bin/env python3
import argparse, os, re, json, random
import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]*\.?\d+)")


def reward_correct(completions, gold, **kw):
    out = []
    for c, g in zip(completions, gold):
        ms = ANS_RE.findall(c)
        if not ms:
            out.append(0.0)
            continue
        try:
            v = float(ms[-1].replace(",", ""))
        except ValueError:
            out.append(0.0)
            continue
        out.append(1.0 if abs(v - g) < 1e-6 else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=32)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--vllm-mem", type=float, default=0.2)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--save-steps", type=int, default=60)
    ap.add_argument("--fewshot-frac", type=float, default=0.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dseed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.dseed)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for r in ds:
        q = r["question"].strip()
        g = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        p = ("<start_of_turn>user\n" + PROMPT_TEMPLATE.format(prompt=q)
             + "<end_of_turn>\n<start_of_turn>model\n")
        rows.append({"prompt": p, "gold": g})
    rng.shuffle(rows)
    train = Dataset.from_list(rows)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.num_gen,
        generation_batch_size=args.gen_batch,
        max_prompt_length=None,
        max_completion_length=args.max_completion,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        loss_type="dr_grpo",
        scale_rewards="none",
        mask_truncated_completions=True,
        vllm_enable_sleep_mode=True,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_max_model_length=1280,
        report_to=[],
        seed=args.dseed,
        model_init_kwargs={"dtype": (torch.float32 if args.dtype == "fp32" else torch.bfloat16),
                           "attn_implementation": "flash_attention_2"},
    )

    trainer = GRPOTrainer(model=args.model, reward_funcs=reward_correct,
                          args=cfg, train_dataset=train)
    # our model terminates with <end_of_turn> (106), not the tokenizer's <eos> (1);
    # TRL's truncation detection compares against trainer.eos_token_id
    trainer.eos_token_id = 106
    from transformers import GenerationConfig
    trainer.model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
