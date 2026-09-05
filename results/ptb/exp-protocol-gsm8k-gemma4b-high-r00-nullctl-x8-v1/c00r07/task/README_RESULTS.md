# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` = uniform weight average of `ckpt/sft2` and `ckpt/sft3`, greedy decoding.

## Results (inspect_evals/gsm8k, 10-shot, `match(numeric=True)`)

| model | decoding | n=150 | n=200 | n=500 | n=1000 |
|---|---|---|---|---|---|
| base `gemma-3-4b-pt` | sampled | **6.0** | | | |
| sft1 | sampled | | 59.0 | | |
| sft2 | sampled | | 66.0 | | |
| sft2 | greedy | | 73.0 | 70.8 | 69.6 |
| sft3 | greedy | | 72.0 | 67.6 | |
| soup(sft2,sft3) | greedy | **75.3** | | 71.2 | 70.6 |

Base model's 6% is a formatting artifact: it never emits a stop token and runs to
`max_tokens`, so the `location="end"` matcher never fires.

## Pipeline

1. **Data** (all decontaminated against the GSM8K test set with `contamination_check.py`,
   0 matches on every file):
   - GSM8K **train** split (7,473) — `####` markers and `<<calc>>` annotations stripped.
   - MetaMathQA `GSM_AnsAug` + `GSM_Rephrased`, re-verified against GSM8K-train gold
     answers (154,422 kept).
   - `nvidia/OpenMathInstruct-2` `augmented_gsm8k` **questions only** (73,396 unique),
     used as rejection-sampling prompts with their `expected_answer` as the verifier.
2. **sft1** — full fine-tune from base on GSM8K-train ×2 + 30k MetaMath (1 epoch,
   lr 1.2e-5, eff. batch 128). → 59%.
3. **RFT round 1** — sft1 samples k=6 @ T=0.9 over 31,473 questions; keep only chains whose
   final answer matches gold; dedupe by numeric skeleton, cap per question.
   Pass rate 59.5%, 83k solutions kept.
4. **sft2** — retrain **from base** on RFT-1 + original GSM8K + 8k MetaMath (70k examples).
   → 66% sampled, **73% greedy**.
5. **RFT round 2** — same with sft2 (pass rate 64.8%) over fresh OMI2 questions; **sft3**
   scored slightly worse, so the two were weight-averaged into the final soup.

## Two things that mattered a lot

- **Greedy decoding: +7 points.** The harness sends no `temperature`, so vLLM falls back to
  the checkpoint's `generation_config.json`; Gemma-3 ships `do_sample: true / top_p 0.95 /
  top_k 64`, i.e. T=1.0 sampling. `set_greedy.py` rewrites it to `temperature: 0.0`.
- **vLLM drops the extra EOS with `n>1`.** Child requests are fanned out from the
  pre-`update_from_generation_config` params, so `<end_of_turn>` was not a stop token and
  every rejection sample ran to `max_tokens` (0% pass rate). Fixed by passing
  `stop_token_ids` explicitly. Related: `LLM.generate()` on a chat-template string
  double-adds `<bos>`; pass token ids instead.

## Files

`prepare_data.py` `gen_rft.py` `build_mix.py` `build_mix3.py` `train_sft.py` `soup.py`
`set_greedy.py` `verify_final.py`; logs and metrics under `runs/`.
