# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` = SFT + GRPO checkpoint (rev `cc012e0a…` as the only starting point).

## Results (inspect_evals/gsm8k via `evaluate.py`)

| model | decoding | n | accuracy |
|---|---|---|---|
| base `gemma-3-4b-pt` | inherited (T=1.0) | 150 | 3.3% |
| SFT v1 | inherited (T=1.0) | 200 | 66.5% |
| SFT v1 | greedy | 200 | 76.5% |
| SFT + GRPO @200 steps | greedy | 500 | 78.2% |
| **SFT + GRPO @300 steps → `final_model`** | **greedy** | **500** | **80.2%** |
| SFT + GRPO @370 steps | greedy | 500 | 79.6% |
| `final_model` under `evaluate.py` defaults | greedy | 150 | 76.7% |

## Pipeline

1. **SFT** (`prep_data.py`, `train_sft.py`) — 118k examples: OpenMathInstruct-2
   `gsm8k` + `augmented_gsm8k` solutions (≤4 per problem), 20k `math` samples for
   diversity, and the original GSM8K **train** CoTs. Every target is rewritten into
   the exact harness format (`…\n\nANSWER: <n>`), 15% of prompts carry a random
   2–5-shot prefix so the model is robust to the harness's 10-shot system message.
   Full-parameter tuning of the language model (vision tower frozen), fp32 master
   weights + 8-bit Adam, liger fused linear-CE (gemma3's 262k vocab OOMs otherwise),
   lr 1.2e-5 cosine, 1 epoch, ~2h on one H100.
2. **Decoding** — the harness sends no sampling params, so vLLM falls back to the
   model's `generation_config.json` (T=1.0/top_p 0.95/top_k 64). Switching the
   shipped config to greedy (`top_k: 1`) is worth **+10 points**. Verified first on
   held-out GSM8K *train* problems (`internal_eval.py`), then on the benchmark.
3. **GRPO** (`train_grpo.py`) — 350 steps, 16 prompts × 8 samples/step, reward =
   exact numeric match of the `ANSWER:` line + small format term, `dr_grpo`, no KL,
   lr 2e-6, vLLM colocate. Prompts from GSM8K train + augmented GSM8K, 25% with a
   fixed k-shot prefix. Training reward 0.71 → 0.78.

Two non-obvious fixes were needed: TRL compares the last completion token against
`tokenizer.eos_token_id` (`<eos>`=1) but gemma3 ends turns with `<end_of_turn>`=106,
so every completion was flagged truncated and masked out (zero gradient); and
`do_sample=false` + `temperature` fails `GenerationConfig.save_pretrained`
validation, hence `top_k: 1` for greedy.

## Decontamination

`contamination_check.py` against `test_data.json`: **0 matches** over the 200k-row
SFT pool and the 15.5k-problem RL prompt set. All problems derive from the GSM8K
**train** split or OpenMathInstruct-2 augmentations of it; the test split was never
read for training.
