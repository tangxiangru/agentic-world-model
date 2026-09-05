# GSM8K post-training of `google/gemma-3-4b-pt`

Start: local snapshot `models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d`.
Final artifact: `final_model/` (full-weight fine-tune, `Gemma3ForConditionalGeneration`, bf16).

## Results (evaluate.py, inspect-ai `inspect_evals/gsm8k`, 10-shot, greedy)

| model | n=150 | n=200 | n=300 | n=500 |
|---|---|---|---|---|
| base gemma-3-4b-pt | 6.7% | | | |
| stage 1 SFT (`runs/sft_v1`) | | 69.5% (sampling) / 70.5% (greedy) | | |
| stage 2 rejection sampling (`runs/rft_v1`) | | | 71.0% | |
| stage 3 GRPO, step 540 → **final_model** | | | **76.0%** | **74.6%** |
| stage 3b GRPO continued, step 684 (rejected) | | | 74.3% | 73.8% |

Base 6.7% → final 74.6% (n=500, stderr 1.9).

## Pipeline

1. **SFT** (`prep_data.py`, `train_sft.py`) — 120k examples from
   `nvidia/OpenMathInstruct-2` (`gsm8k` + `augmented_gsm8k` subsets, 8% `math`/`augmented_math`),
   max 2 solutions per problem, integer answers only, `\boxed{}` stripped and replaced by a
   trailing `ANSWER: <n>` line. Prompts use the exact inspect-ai GSM8K template; 30% of examples
   carry a 1–10-shot prefix drawn from the GSM8K **train** split so the model is robust to the
   10-shot eval prompt. 1 epoch, bs 32, lr 1e-5 cosine, bf16, Liger fused linear CE,
   vision tower frozen. ~1.6 h on one H100.
2. **Rejection sampling / RFT** (`gen_rft.py`, `build_rft_data.py`) — 12 samples for each of the
   7473 GSM8K train problems + 2 samples for 20k `augmented_gsm8k` problems at T=1.0
   (129,676 generations, 68% correct). Kept correct, non-hedging solutions, deduped by numeric
   signature, ≤4 per train problem / ≤1 per augmented problem; off-policy OMI-2 solutions added
   for the 345 train problems never solved. 43,770 examples, 1 epoch at lr 6e-6.
3. **GRPO** (`train_grpo.py`, TRL 0.27, vLLM colocate) — GSM8K train prompts (20% with the exact
   10-shot eval prefix), 8 generations/prompt, 64 completions per optimizer step, `beta=0`
   (no reference model), DAPO loss, group-scaled rewards, lr 1.5e-6 constant, 8-bit AdamW.
   Reward = 1.0 for a correct final answer, −0.1 if `ANSWER:` is missing. 540 steps (~3 h);
   mean training reward rose 0.68 → 0.75. A further 144 steps did not improve the benchmark and
   were discarded.

Note: TRL marks a completion as truncated when its last token is `tokenizer.eos_token_id`. Gemma
stops on `<end_of_turn>` (106), not `<eos>` (1), so `tok.eos_token` is set to `<end_of_turn>`
before constructing `GRPOTrainer`; otherwise every completion is treated as truncated and the
gradient is identically zero.

`final_model/generation_config.json` pins greedy decoding (`temperature: 0.0`), which vLLM picks
up as the server-side default; measured slightly above the model's default sampling config and
removes run-to-run variance.

## Decontamination

All training data was checked with `../contamination_check.py` against `../test_data.json`:
0 contaminated documents for both the 120k stage-1 set and the 43.8k stage-2 set. No GSM8K test
question or answer was used, and no training item was derived from a test item — problems come
from the GSM8K **train** split and from OpenMathInstruct-2 (itself built only from GSM8K/MATH
training splits). GRPO rewards were computed against GSM8K train labels only.

## Diagnostics

`devset.py` measures accuracy off-line under varying shot counts. On 1000 GSM8K train problems the
stage-2 model scored 73.2% (0-shot), 75.9% (1-shot), 76.9% (5-shot), 76.8% (10-shot) — the eval's
10-shot prompt is not a handicap, so no prompt-format work was needed.
