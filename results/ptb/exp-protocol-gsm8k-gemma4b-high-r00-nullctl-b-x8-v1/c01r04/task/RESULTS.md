# GSM8K post-training of `google/gemma-3-4b-pt`

Start: snapshot `cc012e0a6d0787b4adcc0fa2c4da74402494554d` (loaded from the local HF cache; never
re-downloaded, never drifted). `final_model` = weight average of the run2 and run3 checkpoints.

## Results (inspect-ai `inspect_evals/gsm8k`, 10-shot, greedy)

| model | n=150 (default args) | n=300 | n=600 |
|---|---|---|---|
| base `gemma-3-4b-pt` | 8.0 | | |
| run1 (SFT, 1 epoch) | | 78.5 (n=200) | |
| run2 (run1 + 2nd stage) | | 81.3 | 78.5 |
| run3 (run2 + 3rd stage) | | 81.0 | |
| soup(run1,run2) | | 81.0 | |
| soup(run1,run2,run3) .25/.375/.375 | | 84.0 | 80.5 |
| **soup(run2,run3) = final_model** | **82.0** | | **81.8** |
| soup(run4,run2,run3) .5/.25/.25 | | | 80.7 |

`n=150` is the harness default (`python evaluate.py --model-path final_model`), run with all
default arguments (max-connections 2, max-tokens 4000, gpu-memory-utilization 0.3) — exit code 0.

## Approach

**Data.** All training data is public math-instruction data derived from *training* splits only:
- `nvidia/OpenMathInstruct-2` — subsets `gsm8k`, `augmented_gsm8k`, `math`, `augmented_math`
  (≤4 distinct solutions per problem, `\boxed{}` unwrapped, non-numeric / code / LaTeX-heavy
  solutions dropped).
- `meta-math/MetaMathQA` — `GSM_Rephrased` + `GSM_AnsAug` (derived from the GSM8K *train* split).
- Few-shot prefixes built from the GSM8K **train** split, matching `sample_to_fewshot`.

Every example is rendered into the exact eval format: the inspect-ai `MATH_PROMPT_TEMPLATE` as the
user turn, `templates/gemma3.jinja` as the chat template, and a target ending in `ANSWER: <n>`.
12% of examples carry a random *k*-shot system prefix (k ∈ {2,4,10}) so the model is not derailed
by the harness's 10-shot prefix. Loss is masked to the completion only.

**Decontamination.** Every training file was checked with `../contamination_check.py` against
`../test_data.json`: 164,603 + 133,400 + 141,642 documents scanned, **0 contaminated, 0 matches**.

**Training.** Full fine-tune (vision tower + projector frozen, 3.88B trainable), fp32 master
weights with bf16 autocast, 8-bit AdamW, Liger fused-linear-CE, gradient checkpointing,
length-grouped batching, effective batch 128, cosine schedule.
- run1: 160k examples (65M tokens), lr 1e-5, from base
- run2: 144k fresh examples (59M tokens), lr 6e-6, from run1
- run3: 73k fresh examples (29M tokens), lr 5e-6, from run2

**Inference config.** `generation_config.json` in `final_model` sets greedy decoding
(`do_sample: false`, `temperature: 0.0`, no top-k/top-p). vLLM picks these up as default sampling
params; this was worth ~+4 points over the base config's `top_k=64, top_p=0.95` sampling
(78.5 vs 74.5 on the same checkpoint).

**Checkpoint averaging.** Uniformly averaging consecutive fine-tuning checkpoints gave the largest
single late-stage gain: run2 78.5 → soup(run2,run3) 81.8 at n=600 on identical samples.

## Files
- `prep_data.py`, `prep_metamath.py` — dataset construction
- `train_sft.py` — SFT trainer; `soup.py` — weight averaging
- `gen_rft.py` — rejection-sampling data generation (built but abandoned: too slow to be worth the GPU time)
- `run_eval.sh`, `logs/` — evaluation driver and all eval outputs
