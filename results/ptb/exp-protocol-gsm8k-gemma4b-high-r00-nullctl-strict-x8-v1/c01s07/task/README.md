# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` holds the submitted model (full weights, `Gemma3ForConditionalGeneration`, bf16).

## Results (inspect `gsm8k`, 10-shot, via `evaluate.py`)

| model | decoding | limit | accuracy |
|---|---|---|---|
| base `gemma-3-4b-pt` @ `cc012e0a` | sampling | 150 | 0.020 |
| `ckpt/sft_v1` (SFT) | sampling | 200 | 0.600 |
| `ckpt/sft_v1` (SFT) | greedy | 200 | 0.690 |
| `ckpt/m_400` (SFT + GRPO-1) | greedy | 200 | 0.720 |
| `ckpt/rft_v1` (+ rejection-sampling SFT) | greedy | 200 / 500 | 0.745 / 0.732 |
| **`ckpt/g2_200` = `final_model`** (+ GRPO-2) | greedy | 200 / 500 / 1000 | **0.750 / 0.744 / 0.723** |
| `ckpt/g3_*` (GRPO-3, discarded) | greedy | 500 | 0.716 |

`evaluate.py` with its default arguments on `final_model`: **0.733** (limit 150).

## Pipeline

1. **SFT** (`prep_data.py` → `train_sft.py`). 65k GSM8K-style CoT examples drawn from
   `nvidia/OpenMathInstruct-2` (`gsm8k` + `augmented_gsm8k`, plus 8k MATH-style rows for
   diversity), all derived from *training* splits only. Solutions are un-`\boxed`-ed and
   rewritten to end in `ANSWER: <n>`, and prompts reproduce the eval's prompt template and
   `templates/gemma3.jinja` turn format byte-for-byte. 12% of rows carry a few-shot system
   prefix (2–10 GSM8K *train* exemplars) so the model is robust to the eval's 10-shot prefix.
   Full-parameter, 1 epoch, lr 1e-5, vision tower frozen, liger fused-linear-CE.
2. **GRPO round 1** (`train_grpo.py`). LoRA r=64 on the language tower, colocated vLLM
   rollouts, 8 samples/prompt, reward = final `ANSWER` matches the reference. 400 steps.
3. **Rejection-sampling SFT** (`gen_all.py` → `build_sc_data.py --mode rft` → `train_sft.py`).
   6 samples at T=1.0 for 22.5k training problems, keep ≤2 distinct correct solutions per
   problem (36.9k rows), 1 epoch at lr 5e-6 from the GRPO-1 model.
4. **GRPO round 2**. Same recipe, but prompts restricted to a difficulty-filtered set
   (`data/grpo_hard.jsonl`: problems the policy solves 1–5 times out of 6, plus 15% easy and
   25% unsolved) so fewer groups have zero advantage. 200 steps → `final_model`.

A third GRPO round was run and rejected: it scored 0.716 vs 0.744 on the same 500 items.

## Decoding defaults

vLLM seeds its default sampling params from the model's `generation_config.json`. The base
config leaves `temperature` unset, so the eval sampled at T=1.0 — worth ~9 accuracy points
versus greedy on this task (0.600 → 0.690 for the SFT model). `final_model`'s
`generation_config.json` therefore ships `"temperature": 0.0`.

## Decontamination

Every training corpus was checked against the provided GSM8K test copy with
`../contamination_check.py`: 0 contaminated documents out of 65,423 (SFT) and 36,878 (RFT).
Only GSM8K *train* problems and OpenMathInstruct-2 (itself built from training splits) were
used; no test question, answer, or derivative entered training.

## Files

- `prep_data.py` — build the SFT corpus from OpenMathInstruct-2.
- `train_sft.py` — full-parameter SFT / RFT trainer.
- `gen_all.py`, `gen_samples.py` — vLLM rejection sampling from a checkpoint.
- `build_sc_data.py` — turn dumped samples into RFT (or self-consistency) SFT data.
- `train_grpo.py` — LoRA GRPO with colocated vLLM.
- `merge_lora.py`, `set_gen_config.py` — adapter merge and decoding defaults.
- `runs/` — logs and metric JSONs for every experiment above.
