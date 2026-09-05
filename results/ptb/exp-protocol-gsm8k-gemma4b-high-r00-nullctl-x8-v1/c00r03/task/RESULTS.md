# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` = checkpoint **v3** (bit-identical to `ckpt/v3`).

## Results (inspect_evals/gsm8k, 10-shot, greedy)

| model | full test set (1319) | 200 samples |
|---|---|---|
| base gemma-3-4b-pt | – | 6.0% |
| v1 pilot (50k, 1 ep) | – | 71.0% |
| v2 (193k, 1.5 ep) | 79.00% ± 1.12 | 80.0% |
| **v3 = v2 + RFT (1 ep)** | **79.98% ± 1.10** | 84.0% |
| v4 = v2 + RFT + fresh OMI (1 ep) | 78.54% ± 1.13 | – |
| soup(v3, v4) | 79.15% ± 1.12 | – |
| soup(v2, v3, v4) | 79.61% ± 1.11 | – |

`python evaluate.py` with stock defaults on `final_model`: **81.3%** (150 samples).

## Recipe

1. **Data** — OpenMathInstruct-2 (`train_1M` + 6 shards of the full split), restricted to
   `gsm8k` / `augmented_gsm8k` plus some `math` for generality. Solutions reformatted to
   drop `\boxed{}` and end in `ANSWER: <x>`; trained in the exact eval prompt format
   (`templates/gemma3.jinja`, inspect's `MATH_PROMPT_TEMPLATE`), loss on the completion only.
   12% of examples carry a random 2–10-shot GSM8K-*train* system prefix so the model is
   robust to the eval's 10-shot prompt. Every file passed `contamination_check.py`
   against the GSM8K test set with 0 matches.
2. **v2** — full fine-tune from the pinned base snapshot: 193k examples, 1.5 epochs,
   lr 1e-5 cosine, effective batch 64, fp32 master weights + bf16 autocast, 8-bit AdamW,
   gradient checkpointing, Liger fused linear-CE (the 262k vocab makes plain CE the
   memory bottleneck). ~2h32 on one H100.
3. **v3** — rejection-sampling fine-tune: sampled k=4 @ T=1.0 from v2 over 20,473
   GSM8K-train + augmented questions (pass@4 = 96.4%), kept 38,535 verified-correct
   on-policy solutions, then 1 more epoch from v2 on those + 40k OMI at lr 7e-6.

## Two findings that mattered most

- **Decoding config is worth ~4.5 points.** inspect/vLLM send no temperature, so vLLM falls
  back to the model's `generation_config.json` — the base one samples at T=1.0. Writing a
  greedy config into the model dir took the pilot from 66.5% → 71.0%. (`set_gencfg.py`)
- **vLLM's *offline* API does not treat `<end_of_turn>` (106) as a stop token**, only
  `<eos>` (1). The first rejection-sampling run looked like a 6% pass rate purely because
  almost every sample ran to the token cap and was discarded. `stop_token_ids=[1, 106]`
  fixed it (96.4% pass@4). The served path used by `evaluate.py` is unaffected.

## Notes

- The base model was never the bottleneck at 6%: it already solved problems correctly but
  never emitted a stop token, so `match(location="end")` scored a later stray number.
- v2/v3/v4 and the two soups all land in 78.5–80.0%, i.e. within about one standard error
  of each other. v3 is the nominal best and was selected on that basis; the honest read is
  that everything after v2 is inside the noise band.
