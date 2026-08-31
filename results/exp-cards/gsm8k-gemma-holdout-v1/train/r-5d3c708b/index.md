# r-5d3c708b — reconstructed experiment cards

Base model post-trained: Qwen/Qwen3-4B-Base · benchmark: gsm8k · budget: 10 h on one H100 80GB.
13 cards, one per launch. Baseline (base model, `--limit 60`) = 0.3833 at [80]; not a card (no candidate produced).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 113 | 0.18 | sft | base_model | work/gsm8k_train.jsonl (openai/gsm8k train, cleaned) | 1e-5 / 3 | completed | 0.800 @ n=150 (logs/iter1.json) | inconclusive | reject |
| exp-02 | 287 | 1.25 | rft | base_model | work/gsm8k_train.jsonl + work/rft_all.jsonl (synthetic:self from exp-01) | 1e-5 / 3 | completed | 0.807 @ n=150 (logs/iter2.json) | inconclusive | adopt |
| exp-03 | 382 | 2.83 | grpo | exp-02 | openai/gsm8k train (in-process, verifiable reward) | 1e-6 / 600 steps | failed | — (crashed at the step-150 save) | inconclusive | abandon_line |
| exp-04 | 420 | 3.08 | grpo | exp-02 | openai/gsm8k train (in-process, verifiable reward) | 1e-6 / 600 steps | completed | 0.920 @ n=300, checkpoint-600 (logs/eval_work_grpo_checkpoint-600.json) | inconclusive | adopt |
| exp-05 | 474 | 4.22 | other (package → final_model) | exp-04 | — | — | completed | — (fp32 copy, never scored) | inconclusive | reject |
| exp-06 | 497 | 4.57 | other (bf16 → final_model) | exp-04 | — | — | failed | — (save rejected; final_model wiped) | inconclusive | abandon_line |
| exp-07 | 503 | 4.58 | other (bf16 → final_model) | exp-04 | — | — | completed | 0.8817 @ n=1319 (logs/final_mc2.json) | inconclusive | adopt |
| exp-08 | 546 | 4.99 | grpo | exp-07 | openai/gsm8k train (in-process, verifiable reward) | 1e-6 / 600 steps | completed | 0.9000 @ n=1319, ckpt-400 (logs/val_work_g2_400.json) | supported | adopt |
| exp-09 | 614 | 6.99 | other (package → final_model) | exp-08 | — | — | completed | 0.940 @ n=150, grader config (logs/final_confirm.json) | inconclusive | reject |
| exp-10 | 624 | 7.05 | grpo (beta=0.04) | exp-08 | openai/gsm8k train (in-process, verifiable reward) | 1e-6 / 300 steps | failed | — (OOM at startup) | inconclusive | abandon_line |
| exp-11 | 636 | 7.15 | grpo | exp-08 | openai/gsm8k train (in-process, verifiable reward) | 1e-6 / 120 steps | completed | 0.9075 @ n=1319, ckpt-40 (logs/val_g3_40.json) | supported | adopt |
| exp-12 | 671 | 8.07 | other (package → final_model) | exp-11 | — | — | completed | 0.940 @ n=150, grader config (logs/final_confirm2.json) | inconclusive | adopt |
| exp-13 | 681 | 8.12 | other (config: torch_dtype) | exp-12 | — | — | completed | 0.9037 @ n=1319 and 0.9467 @ n=150, grader config (logs/FINAL_full.json, logs/final_confirm3.json) | inconclusive | adopt |

**Submitted candidate:** exp-13 — `final_model` = exp-11's grpo3 checkpoint-40, cast to bf16, greedy generation config (eos `<|im_end|>` 151645), `torch_dtype: bfloat16`.

**Chain:** base → exp-01 (SFT) → [samples the RFT data] → exp-02 (SFT+RFT, from base) → exp-03/exp-04 (GRPO) → exp-05/06/07 (packaging, fp32 → bf16) → exp-08 (continued GRPO) → exp-09 (packaging) → exp-10/exp-11 (third GRPO) → exp-12 → exp-13.

Not cards: the baseline eval at [46]; `generate_rft.py` data-generation runs (recorded as `setup.data[].build_command` on exp-02, with the two `--limit 64` dry runs as `provenance.smoke_runs`); the two 3-step GRPO smoke tests at [361]/[369] (`provenance.smoke_runs` on exp-03); checkpoint sweeps and validation evals (recorded under `evaluation` / `result.measurements`); `to_bf16.py` dtype casts into `work/` (same weights, recorded on the card that produced the source checkpoint).
