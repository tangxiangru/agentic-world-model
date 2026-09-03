#!/usr/bin/env python3
"""GRPO RL on GSM8K train split (gold answers) starting from the SFT checkpoint."""
import argparse, json, os, re
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

PROMPT = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.\n\n'
    "{q}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) '
    "where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.\n\n"
    "Reasoning:"
)

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_num(t):
    ms = NUM.findall(t.replace("$", "").replace("*", ""))
    if not ms:
        return None
    v = ms[-1].replace(",", "").rstrip(".")
    try:
        return float(v)
    except ValueError:
        return None


def reward_correct(completions, answer, **kw):
    out = []
    for c, gold in zip(completions, answer):
        txt = c[0]["content"] if isinstance(c, list) else c
        g = last_num(str(gold))
        if "ANSWER:" in txt:
            p = last_num(txt.split("ANSWER:")[-1])
        else:
            p = last_num(txt)
        ok = (p is not None and g is not None and abs(p - g) < 1e-4)
        out.append(1.0 if ok else 0.0)
    return out


def reward_format(completions, **kw):
    out = []
    for c in completions:
        txt = c[0]["content"] if isinstance(c, list) else c
        out.append(0.1 if "ANSWER:" in txt else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default="runs/grpo_v1")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--ngen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--vllm-gpu", type=float, default=0.22)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.init)
    with open("templates/gemma3.jinja") as f:
        tok.chat_template = f.read()
    # the chat-format terminator is <end_of_turn> (106), not <eos> (1); TRL uses
    # tokenizer.eos_token_id for both stopping and termination accounting.
    tok.eos_token = "<end_of_turn>"
    print("eos_token_id:", tok.eos_token_id)

    d = load_dataset("openai/gsm8k", "main")["train"]
    rows = []
    for r in d:
        rows.append({
            "prompt": [{"role": "user", "content": PROMPT.format(q=r["question"])}],
            "answer": r["answer"].split("####")[1].strip().replace(",", ""),
        })
    ds = Dataset.from_list(rows).shuffle(seed=0)
    print("train prompts:", len(ds))

    try:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3
        apply_liger_kernel_to_gemma3(fused_linear_cross_entropy=False)
        print("liger applied")
    except Exception as e:
        print("liger failed", e)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_grad_norm=0.5,
        adam_beta2=0.99,
        weight_decay=0.0,
        bf16=True,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.ngen,
        max_prompt_length=384,
        max_completion_length=args.max_completion,
        temperature=args.temp,
        top_p=1.0,
        beta=args.beta,
        loss_type="dr_grpo",
        scale_rewards=False,
        epsilon=0.2,
        epsilon_high=0.28,
        num_iterations=1,
        max_steps=args.steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=6,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu,
        vllm_max_model_length=1024,
        log_completions=False,
        report_to=[],
        seed=0,
    )
    trainer = GRPOTrainer(
        model=args.init,
        reward_funcs=[reward_correct, reward_format],
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    import shutil
    for f in ["preprocessor_config.json", "processor_config.json", "generation_config.json",
              "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json"]:
        s = os.path.join(BASE, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(args.out, f))
    print("saved", args.out)


if __name__ == "__main__":
    main()
