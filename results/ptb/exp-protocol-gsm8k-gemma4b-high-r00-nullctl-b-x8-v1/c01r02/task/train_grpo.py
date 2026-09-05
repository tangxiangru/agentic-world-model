#!/usr/bin/env python3
"""GRPO on GSM8K-train-derived problems: reward = final answer matches gold."""
from __future__ import annotations
import argparse, glob, json, os, random, re, sys
import torch
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer, Gemma3ForCausalLM
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_data import USER_TMPL, norm_answer, is_plain_number, build_fewshot_pool

BOS, SOT, EOT = '<bos>', '<start_of_turn>', '<end_of_turn>'
ANS_RE = re.compile(r'ANSWER:\s*(.+?)\s*$', re.MULTILINE)
RNG = random.Random(5)


def extract(text):
    m = ANS_RE.findall(text)
    return norm_answer(m[-1]) if m else None


def correct(pred, gold):
    if pred is None:
        return False
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        return False


def reward_correct(completions, answer, **kw):
    return [1.0 if correct(extract(c), a) else 0.0 for c, a in zip(completions, answer)]


def reward_format(completions, **kw):
    return [0.1 if ANS_RE.search(c) else 0.0 for c in completions]


def build_dataset(n_aug, fewshot_frac):
    rows = []
    f = sorted(glob.glob('/home/ben/hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet'))[0]
    for r in pq.read_table(f).to_pylist():
        _, ans = r['answer'].split('####')
        rows.append((r['question'].strip(), norm_answer(ans)))
    if n_aug:
        seen = {}
        for fp in sorted(glob.glob('/home/ben/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2/snapshots/*/data/train_1M-*.parquet')):
            for r in pq.read_table(fp).to_pylist():
                if r['problem_source'] != 'augmented_gsm8k':
                    continue
                if not is_plain_number(r['expected_answer']) or len(r['problem']) > 1200:
                    continue
                seen.setdefault(r['problem'].strip(), norm_answer(r['expected_answer']))
        aug = list(seen.items())
        RNG.shuffle(aug)
        rows += aug[:n_aug]
    RNG.shuffle(rows)

    pool = build_fewshot_pool(None)
    data = []
    for q, a in rows:
        user = USER_TMPL.format(prompt=q)
        if RNG.random() < fewshot_frac:
            k = RNG.choice([1, 2, 3, 4, 10])
            user = '\n\n'.join(RNG.sample(pool, k)) + '\n\n' + user
        data.append({'prompt': f"{BOS}{SOT}user\n{user}{EOT}\n{SOT}model\n", 'answer': a})
    return Dataset.from_list(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='runs/sft1/final')
    ap.add_argument('--out', default='runs/grpo')
    ap.add_argument('--n-aug', type=int, default=20000)
    ap.add_argument('--fewshot-frac', type=float, default=0.12)
    ap.add_argument('--num-gen', type=int, default=8)
    ap.add_argument('--gen-batch', type=int, default=256)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--max-completion', type=int, default=512)
    ap.add_argument('--max-prompt', type=int, default=2300)
    ap.add_argument('--vllm-util', type=float, default=0.24)
    ap.add_argument('--save-steps', type=int, default=50)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    # The chat format terminates turns with <end_of_turn> (106), not <eos> (1).
    # TRL uses processing_class.eos_token_id both as the vLLM stop token and to
    # decide whether a completion terminated; without this every rollout counts
    # as truncated and gets masked out.
    tok.eos_token = '<end_of_turn>'
    assert tok.eos_token_id == 106, tok.eos_token_id
    ds = build_dataset(args.n_aug, args.fewshot_frac)
    # vLLM rejects prompts longer than max_model_len, so drop them up front
    # (TRL's max_prompt_length truncation is not applied to the rollout prompts).
    n0 = len(ds)
    ds = ds.filter(lambda x: len(tok(x['prompt'], add_special_tokens=False)['input_ids']) <= args.max_prompt,
                   num_proc=8)
    print('grpo dataset', len(ds), 'dropped', n0 - len(ds))

    model = Gemma3ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    model.config.use_cache = False

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type='constant_with_warmup',
        warmup_steps=10,
        adam_beta2=0.95,
        max_grad_norm=0.5,
        optim='adamw_bnb_8bit',
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.gen_batch // args.bs,
        generation_batch_size=args.gen_batch,
        num_generations=args.num_gen,
        num_iterations=1,
        beta=0.0,
        epsilon=0.2,
        loss_type='dr_grpo',
        scale_rewards='none',
        mask_truncated_completions=True,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        use_vllm=True,
        vllm_mode='colocate',
        vllm_gpu_memory_utilization=args.vllm_util,
        vllm_max_model_length=args.max_prompt + args.max_completion + 16,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=12,
        report_to=[],
        log_completions=False,
        seed=0,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_correct, reward_format],
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out + '/final')
    import shutil
    for fn in ('tokenizer.json', 'tokenizer.model', 'tokenizer_config.json',
               'special_tokens_map.json', 'added_tokens.json', 'generation_config.json'):
        src = os.path.join(args.model, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, 'final', fn))
    print('saved', args.out + '/final')


if __name__ == '__main__':
    main()
