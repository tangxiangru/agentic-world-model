#!/usr/bin/env python3
"""GRPO on GSM8K *train* problems, starting from the SFT checkpoint."""
import argparse, json, os, re, sys
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def to_float(s):
    try:
        return float(str(s).replace(",", "").rstrip("."))
    except Exception:
        return None


def extract_answer(text):
    m = re.findall(r"ANSWER:\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return to_float(m[-1])
    m = NUM.findall(text)
    return to_float(m[-1]) if m else None


def norm_q(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def correctness_reward(completions, gold, **kwargs):
    out = []
    for c, g in zip(completions, gold):
        text = c[0]["content"] if isinstance(c, list) else c
        v = extract_answer(text)
        out.append(1.0 if (v is not None and abs(v - g) < 1e-6) else 0.0)
    return out


def format_reward(completions, **kwargs):
    out = []
    for c in completions:
        text = (c[0]["content"] if isinstance(c, list) else c).strip()
        out.append(0.2 if re.search(r"ANSWER:\s*-?[\d,]+\.?\d*\s*$", text) else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="runs/grpo1")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=24)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--vllm-mem", type=float, default=0.22)
    ap.add_argument("--n-aug", type=int, default=0)
    args = ap.parse_args()

    test = json.load(open("/home/ben/test_data.json"))
    test_norm = {norm_q(x["question"]) for x in test}

    rows = []
    gs = load_dataset("openai/gsm8k", "main", split="train")
    for q, a in zip(gs["question"], gs["answer"]):
        if norm_q(q) in test_norm:
            continue
        g = to_float(a.split("####")[-1].strip())
        if g is None:
            continue
        rows.append({"prompt": [{"role": "user",
                                 "content": MATH_PROMPT_TEMPLATE.format(prompt=q.strip())}],
                     "gold": g})
    if args.n_aug:
        import pyarrow.parquet as pq, random
        aug = pq.read_table("data/omi2_augmented_gsm8k.parquet").to_pylist()
        random.Random(11).shuffle(aug)
        seen = set()
        for r in aug:
            if len(seen) >= args.n_aug:
                break
            q = r["problem"].strip()
            k = norm_q(q)
            if k in seen or k in test_norm:
                continue
            g = to_float(r["expected_answer"])
            if g is None:
                continue
            seen.add(k)
            rows.append({"prompt": [{"role": "user",
                                     "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}],
                         "gold": g})
    ds = Dataset.from_list(rows).shuffle(seed=0)
    print("prompts:", len(ds))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = open("templates/gemma3.jinja").read()
    # The SFT model ends its turn with <end_of_turn> (106), not <eos> (1). TRL uses
    # processing_class.eos_token_id to decide which completions actually terminated,
    # so without this every sample looks truncated and gets masked out of the loss.
    tok.eos_token = "<end_of_turn>"
    assert tok.eos_token_id == 106, tok.eos_token_id

    from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text
    apply_liger_kernel_to_gemma3_text(fused_linear_cross_entropy=False)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        max_grad_norm=0.2,
        adam_beta2=0.99,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.gen,
        max_prompt_length=512,
        max_completion_length=args.max_completion,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        loss_type="dapo",

        scale_rewards="group",
        mask_truncated_completions=True,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_max_model_length=1024,
        log_completions=False,
        model_init_kwargs={"dtype": torch.float32,
                           "attn_implementation": "flash_attention_2"},
    )
    trainer = GRPOTrainer(model=args.model, args=cfg,
                          reward_funcs=[correctness_reward, format_reward],
                          train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
