# GSM8K post-training of `google/gemma-3-4b-pt`

Final model: `final_model/` — a full-parameter fine-tune of the immutable snapshot
`cc012e0a6d0787b4adcc0fa2c4da74402494554d` (SFT → GRPO), served greedily.

## Results (GSM8K test via `evaluate.py`, 10-shot prompt, greedy decoding)

| stage | full test (1319) | notes |
|---|---|---|
| base `gemma-3-4b-pt` | — | 2.67% on 150 items; never emits the `ANSWER:` format |
| SFT (`runs/sft_a`) | — | 50.5% sampled / **64.0%** greedy on 200 items |
| + GRPO stage 1 (`runs/grpo_a`, 330 steps) | — | 75.0% on 200 items (ckpt-160) |
| + GRPO stage 2 (`runs/grpo_b`, ckpt-70) | **75.28%** | ← **`final_model`** |
| GRPO stage 2 ckpt-160 | 74.75% | |
| soup of 4 GRPO ckpts | 74.37% | weight averaging did not help |
| soup of 11 GRPO ckpts | 74.22% | |
| ckpt-70 + `repetition_penalty=1.05` | 74.15% | hurt |

`python evaluate.py` with its stock defaults (150 items, `--max-connections 2`) gives 71.3%.

## Pipeline

1. **Data** (`prep_data.py`) — 81,193 examples, all GSM8K-*train*-derived:
   OpenMathInstruct-2 rows with `problem_source ∈ {gsm8k, augmented_gsm8k}` (one solution per
   problem, `\boxed{}` stripped, final-number-consistency filtered) plus the original GSM8K-train
   gold chains-of-thought. Every completion is rewritten into the exact eval format
   (`<reasoning>\n\nANSWER: <number>`), and prompts use the eval's own template verbatim.
   8% of training examples get a random 1–3-shot prefix so the model is robust to the eval's
   10-shot system message.
   Contamination: `contamination_check.py` over all 154,219 unique (question, solution)
   documents → **0 matches** against the test set (`logs/decon.log`).
2. **SFT** (`train_sft.py`) — full-parameter, vision tower frozen, bf16, liger fused
   linear-CE, 1 epoch, lr 1e-5 cosine, effective batch 96, length-bucketed sampler. ~79 min.
3. **GRPO** (`grpo_train.py`) — TRL, colocated vLLM, Dr.GRPO, `beta=0` (no reference model),
   32 prompts × 8 rollouts per update, lr 3e-6 (stage 1) / 2e-6 (stage 2), reward = numeric
   match of the `ANSWER:` line. Stage 1: 330 steps on GSM8K train; train reward 0.57 → 0.78.
   Stage 2: 160 steps on GSM8K train + 20k `augmented_gsm8k` problems; flat, best at step 70.
4. **Selection** — five candidates evaluated on the full 1319-item test set; `runs/grpo_b/checkpoint-70` won.

## Gotchas worth remembering

- `evaluate.py` passes no `temperature`, so vLLM inherits the model's `generation_config.json`.
  Writing `{"do_sample": false, "temperature": 0.0}` there is worth **+13.5 points** over the
  base config's temperature-1.0 sampling.
- Gemma-3 ends turns with `<end_of_turn>` (106), not `<eos>` (1). TRL uses
  `tokenizer.eos_token_id` to detect truncation, so without `tok.eos_token = "<end_of_turn>"`
  every rollout is marked truncated and `mask_truncated_completions=True` zeroes the loss.
- vLLM will not load a Gemma-3 checkpoint without `preprocessor_config.json` /
  `processor_config.json`; Trainer checkpoints don't write them.

## Files

`prep_data.py` · `train_sft.py` · `grpo_train.py` · `soup.py` · `prep_ckpt.py` · `finalize.py` ·
`run_eval.sh` · `gen_samples.py`/`build_rft_data.py` (rejection-sampling path, written but not
used — GRPO was the better spend). Metrics in `logs/*.json`.
