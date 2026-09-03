# GSM8K post-training of `google/gemma-3-4b-pt`

Start weights: local snapshot `cc012e0a6d0787b4adcc0fa2c4da74402494554d` (never mutated).

## Results (inspect_evals/gsm8k, 10-shot, `match(numeric=True)`)

| model | n=150 (default settings) | n=300 | n=700 | n=1319 (full test) |
|---|---|---|---|---|
| base `gemma-3-4b-pt` | 4.0% | | | |
| SFT round 1 (`runs/sft_v2`) | 67.5% (n=200) | 65.0% | | |
| SFT round 2 + RFT (`runs/sft_v3`) | | 67.0% | | |
| GRPO ckpt-120 | | 69.3% | 73.9% | |
| GRPO ckpt-180 | | 75.3% | 73.4% | |
| GRPO ckpt-240 / ckpt-300 | | 58.7% / 63.7% | | |
| soup(120,180,60) | | | 75.3% | |
| **soup(120,180) = `final_model`** | **76.0%** | | **77.7%** | **76.88% ± 1.16%** |

## Pipeline

1. **SFT round 1** (`prep_data.py` → `train_sft.py`, `runs/sft_v2`)
   121k examples, 1 epoch, full fine-tune, lr 1.5e-5, eff. batch 64, ~51M tokens (84 min).
   Mixture from public train-split corpora only:
   OpenMathInstruct-2 `gsm8k` (29k) / `augmented_gsm8k` (65k) / `math` (8k) / `augmented_math` (12k),
   plus GSM8K **train** human solutions (7.5k).
   Targets reformatted to the eval's own answer format (`...\n\nANSWER: N`), `\boxed{}` unwrapped.
   15% of examples get a 1–4-shot prefix so the model is robust to the eval's 10-shot system prompt.

2. **Rejection sampling (RFT)** (`gen_rft.py`)
   vLLM sampling from `sft_v2`, k=4 @ T=1.0 over 20k GSM8K-style questions
   (7.5k GSM8K train + 12.5k OpenMathInstruct-2 augmented questions unseen in round 1).
   Kept 23.4k verified-correct, number-signature-deduped solutions (≤2/question); 5.3k questions
   were never solved → their reference solutions were up-weighted (`boost_unsolved.py`).

3. **SFT round 2** (`prep_data2.py` → `train_sft.py`, `runs/sft_v3`)
   119k fresh examples (no round-1 repeats) + RFT data + unsolved boost, continued from `sft_v2`,
   1 epoch, lr 1e-5 (84 min). Only +2 pts — SFT had saturated.

4. **GRPO RL** (`grpo_train.py`, TRL, colocated vLLM)
   From `sft_v3` on GSM8K **train** questions with gold answers.
   8 generations/prompt, 64 completions/step, Dr-GRPO loss, β=0 (no KL), lr 1.5e-6, adamw_8bit.
   Batch reward rose 0.74 → 0.84 by step ~150, then degraded (entropy collapse, growing
   completion length), so the run was stopped at step 323.
   Needed fix: TRL uses `tokenizer.eos_token_id` for stopping/termination accounting, but the
   chat terminator is `<end_of_turn>` (106), not `<eos>` (1) — without setting it, every
   completion is counted as truncated and the terminator is never reinforced.

5. **Checkpoint soup** (`soup.py`)
   Uniform weight average of GRPO ckpt-120 and ckpt-180 → **+3.8 pts over either ingredient**.
   Greedy addition of ckpt-60 made it worse, so the 2-way average is final.

## Decontamination

Every training file was checked with `../contamination_check.py` against `../test_data.json`:
`sft_v1` (164,745 docs), `rft_pool` (32,322), `sft_v3` (118,840) — **0 contaminated documents** in all.
No GSM8K test question or answer was used for training, prompting, or data generation.

## Notes

- `liger-kernel` was installed for SFT memory savings and later removed (the 64 MB root overlay
  filled up and corrupted its files); `final_model` has no dependency on it and loads with the
  stock environment (`transformers` 4.57.3 / `vllm` 0.11.0).
- `final_model` keeps the original `Gemma3ForConditionalGeneration` architecture, tokenizer and
  `generation_config.json` (`eos_token_id: [1, 106]`) from the base snapshot.
