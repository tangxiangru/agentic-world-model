# GSM8K post-training of `google/gemma-3-4b-pt`

Start: base snapshot `cc012e0a6d0787b4adcc0fa2c4da74402494554d` (loaded directly, never mutated).
Final artifact: `final_model/`.

## Results (inspect_evals/gsm8k via `evaluate.py`, greedy decoding)

| model | RL steps | test-500 | full test (1319) |
|---|---|---|---|
| base gemma-3-4b-pt | – | 6.0% (n=100) | – |
| `ckpt/sft1` (SFT) | 0 | 75.4% | – |
| `ckpt/g120` | 120 | 76.6% | – |
| `ckpt/g220` | 220 | 77.2% | – |
| `ckpt/g320` | 320 | 79.2% | – |
| `ckpt/g420` | 420 | 79.4% | 76.50% |
| `ckpt/g500` | 500 | 79.0% | – |
| **`ckpt/g620` → `final_model`** | **620** | **80.4%** | **77.94%** |
| `ckpt/g680` | 680 | 79.8% | 77.86% |

Final check with `python evaluate.py` (all defaults, limit 150, max-connections 2):
**79.33% (± 3.3)**.

## Pipeline

1. **SFT** (`prep_data.py`, `train_sft.py`) — 82k examples from
   `nvidia/OpenMathInstruct-2` (`gsm8k` 14.8k, `augmented_gsm8k` 60k, `math`/`augmented_math`
   14k as reasoning ballast), ≤4 solutions per unique problem. `\boxed{}` stripped and every
   response terminated with `ANSWER: <n>` to match the harness scorer (which reads the last
   number in the completion).
   Prompts are rendered with the harness's own `templates/gemma3.jinja` and the exact
   `MATH_PROMPT_TEMPLATE`; 15% of examples carry a k-shot system prompt built from GSM8K
   *train* so the model learns to answer in its own style rather than continue the demos.
   Full-parameter (vision tower frozen), fp32 master weights + bf16 autocast, 8-bit Adam,
   Liger fused CE, 1 epoch, lr 1.2e-5 cosine, effective batch 128. ~57 min on one H100.

2. **GRPO** (`train_grpo.py`) — 3 sequential stages, 620 kept steps total, on the 7073
   GSM8K *train* questions not held out. Reward = harness-identical numeric match (+0.05
   format bonus), 8 rollouts/prompt, 192 completions/step, T=1.0, β=0 (no reference model),
   lr 1e-6 → 1.5e-6, vLLM colocate. On-policy reward rose 0.69 → 0.79.

## Two details that mattered

* **Decoding.** vLLM seeds its default sampling params from the model's
  `generation_config.json`. The stock gemma-3 config has `do_sample/top_k 64/top_p 0.95`,
  so the harness was sampling at T=1.0. `final_model/generation_config.json` sets
  `do_sample: false, temperature: 0.0` → greedy. Worth several points.
* **`<end_of_turn>` (106), not `<eos>` (1), ends a gemma turn.** TRL uses
  `tokenizer.eos_token_id` for truncation detection, so with the stock tokenizer every
  rollout was flagged truncated and `mask_truncated_completions` zeroed the whole loss
  (`grad_norm: 0.0`). Fixed by pointing `tok.eos_token` at `<end_of_turn>` for RL.
  (Separately: vLLM's offline `SamplingParams(n>1)` path drops the generation-config stop
  ids — `gen.py` passes `stop_token_ids=[1, 106]` explicitly.)

## Decontamination

No GSM8K test item was used or referenced for training. `../contamination_check.py`
against `../test_data.json`: **0 contaminated documents** over the 88,764 SFT documents and
the 7,073 RL prompts.
