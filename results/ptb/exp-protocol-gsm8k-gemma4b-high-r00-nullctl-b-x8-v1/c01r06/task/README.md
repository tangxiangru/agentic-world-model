# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` — full-parameter fine-tune of the pinned snapshot
`cc012e0a6d0787b4adcc0fa2c4da74402494554d`. Nothing else was fine-tuned.

## Results (`evaluate.py`, 10-shot, `match(numeric=True)`)

| model | full test set (1319) | 400-item subset |
|---|---|---|
| base `gemma-3-4b-pt` | — | 5.0% (n=100) |
| SFT round 1 (`runs/sft1`) | 73.54% ± 1.22 | 74.75% |
| **SFT round 2 = `final_model`** | **74.68% ± 1.20** | **78.0%** |
| soup 0.5·sft1 + 0.5·sft2 | — | 76.0% |

`python evaluate.py` with its default arguments reproduces 74.0% (n=150).

The base model already solved many problems correctly but never stopped — it kept
generating further Q/A pairs, and the scorer reads the *last* `ANSWER:` line. Most
of the jump from 5% is learning the answer format and the stop token.

## Method

1. **Data** (`build_data.py`, `build_data2.py`) — grade-school CoT reformatted to the
   eval's exact prompt/answer shape, ending in a bare `ANSWER: <number>` line:
   - `nvidia/OpenMathInstruct-2`, `problem_source ∈ {gsm8k, augmented_gsm8k}`
     (derived from the GSM8K **train** split), plus a slice of `math`/`augmented_math`
   - original `openai/gsm8k` **train** rationales
   - light LaTeX cleanup, `\boxed{}` stripped, non-numeric answers dropped
2. **Training** (`train_sft.py`) — full fine-tune of the language model (vision tower
   frozen), completion-only loss, bf16, gradient checkpointing, `adamw_torch_fused`,
   cosine-to-10% schedule, effective batch 64.
   - Round 1: 120k rows / 55.9M tokens, lr 2e-5, 1 epoch (1h35m)
   - Round 2: 92k rows / 40.7M tokens from round 1, lr 8e-6, 1 epoch (1h09m)
   - 12% of examples carry a 1–10-shot GSM8K-train prefix so the model is trained
     under the same 10-shot condition the eval uses.
   - `SparseHeadTrainer` applies the 262k-wide LM head only at supervised positions;
     with Gemma-3's vocabulary this is the dominant cost otherwise.
3. **Rejection sampling / RFT** (`gen_samples.py`, `build_rft.py`) — 6 samples per
   GSM8K-train question from round 1 at T=1.0 (71% correct; 94% of questions solved
   at least once), filtered to correct answers and deduplicated by arithmetic
   signature → 13.4k rows, included ×2 in the round-2 mixture.
4. **Decoding defaults** — the harness sends no sampling parameters, so vLLM falls
   back to the model's own `generation_config.json`. The pretrained default is
   temperature-1.0 sampling. `final_model` ships `temperature: 0.0`, which measured
   **+2.75 points** over temperature 0.01 on the same 400 items.
   Two details matter: vLLM clamps any temperature in `(0, 0.01)` **up** to 0.01, so
   `1e-06` is *not* greedy; and `do_sample: true` alongside `temperature: 0.0` is the
   only spelling that is both truly greedy in vLLM and valid for
   `GenerationConfig.save_pretrained`.
   `repetition_penalty: 1.05` was tested and rejected (76.75% vs 78.0%).

## Decontamination

Every training document was checked with `../contamination_check.py` against
`../test_data.json`: **0 contaminated documents** out of 189,767 (v1 pool), 200,000
(v2 pool) and 13,373 (RFT rows). No test question, answer, or anything derived from a
test item was used; the GSM8K *train* split and train-derived public datasets only.

## Things that did not help

- Weight-averaging sft1 and sft2 (76.0% vs 78.0% on the 400-item subset).
- Repetition penalty, despite 1.5% of greedy generations degenerating into loops.
- The RFT round itself was worth ~+1 point on the full set; the SFT corpus is already
  rejection-sampled from a much stronger teacher, so the on-policy gain is small.
