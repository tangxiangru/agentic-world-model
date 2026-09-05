# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` holds the submitted checkpoint.

| stage | GSM8K (full 1319 test items, greedy) |
|---|---|
| base `gemma-3-4b-pt` (10-shot eval prompt) | 4.7% (150-item slice) |
| SFT round 1 | 77.4% (150-item slice: 76.7%) |
| SFT + rejection-sampling data (round 2) | ~75% (150/300-item slices; no gain, discarded) |
| SFT1 + GRPO, 100 steps (`runs/g100`) | 77.41% |
| **+ 75 more GRPO steps @ lr 6e-7 (`runs/h75` = `final_model`)** | **79.30%** |

Default `evaluate.py` invocation (150 items): **80.7 – 81.3%**.

## Pipeline

1. **`make_base.py`** — extracts the text decoder (`Gemma3ForCausalLM`, 3.88B) from the
   immutable `cc012e0a…` snapshot of `gemma-3-4b-pt` into `base_text/`. The SigLIP tower is
   unused for this task; dropping it saves memory and keeps vLLM on the pure-text path.
2. **`extract_omi2.py` / `build_sft.py`** — 73.5k CoT records from
   * `openai/gsm8k` **train** split (original human rationales, `<<…>>` stripped),
   * `nvidia/OpenMathInstruct-2`, `problem_source ∈ {gsm8k, augmented_gsm8k}` (both derived
     from the GSM8K **train** split only).

   Every target is rewritten to end in `ANSWER: <n>` with thousands separators — the
   `match(numeric=True, location="end")` scorer strips numeric punctuation for the 98.8% of
   targets that are `isnumeric()`, but the 16 comma-formatted targets need the literal comma.
3. **`train_sft.py`** — full fine-tune, 1 epoch, fp32 master weights + bf16 autocast,
   8-bit Adam, Liger fused CE (the 262k vocab otherwise OOMs), lr 1e-5, effective batch 144.
   Prompts are rendered with the *exact* eval chat template and user template; 12% of
   examples carry the 10-shot system prefix the eval always uses, 16% carry 1–8 shots, the
   rest are zero-shot, so the model is robust to the eval's in-context prefix.
4. **`gen_rft.py`** — rejection sampling (6 samples @ T=1.0 over 19.5k train problems, 93.3%
   solved) → 35k on-policy records. Retraining from base on the union (`runs/sft2_bf16`) did
   **not** beat SFT1 (McNemar p≈0.54), so this branch was dropped.
5. **`train_grpo.py`** — GRPO (TRL, vLLM colocate, no KL) on the 7.4k GSM8K train problems,
   reward = exact numeric match + small format bonus, 8 completions/prompt, 16 prompts/step.
   Run 1: 350 steps @ lr 1.5e-6 — train reward 0.65 → 0.87, but test accuracy peaks near
   step 100 and then degrades with entropy collapse. Run 2 restarts from step 100 at
   lr 6e-7; its step-75 checkpoint is the best model. Run 3 (50 more steps @ 5e-7) and
   several weight soups were evaluated and did not improve on it.

## Decoding

`final_model/generation_config.json` sets `temperature: 0.0` and `eos_token_id: [1, 106]`.
vLLM takes its default sampling params from this file, so the served model decodes greedily
and stops on `<end_of_turn>`; the released pt config would otherwise sample at T=1.0.

## Decontamination

`../test_data.json` was used only as the reference for `../contamination_check.py`. All
training sources derive from the GSM8K *train* split; questions were additionally filtered
against normalised test questions (0 hits), and both `data/sft.jsonl` and `data/rft1.jsonl`
pass the contamination checker with 0 flagged documents.

## Note on the environment

`networkx/generators/__pycache__/small.cpython-310.pyc` in site-packages was corrupt
("bad marshal data"), which intermittently killed `vllm serve` on startup. It has been
regenerated; if the failure reappears, delete that `.pyc`.
