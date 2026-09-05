# GSM8K post-training of google/gemma-3-4b-pt — batch summary

**Shipped model:** `final_model/` — uniform weight average of exp-02 and exp-04, packaged bf16 with a
greedy `generation_config.json`. Verified by the official harness from that exact path.

**Score:** 0.7582 ± 0.0118 on all 1319 GSM8K test items (`eval/exp-08_final_model_full.json`).
Base checkpoint under the same harness: 0.0867 on dev-150 / 0.78 for the shipped model on the same 150.

| card | intervention | result | decision |
|---|---|---|---|
| exp-01 | baseline measurement of the frozen snapshot | 0.0867 (dev-150); 66/150 completions never stop | iterate |
| exp-02 | completion-only SFT, 99k gsm8k-derived CoT rows from OpenMathInstruct-2 | 0.6667 (sampled decode); non-stopping 66/150 → 0/150 | adopt |
| exp-03 | decode config: temperature 0 in `generation_config.json`, same weights | 0.78 (+11.3) | adopt |
| exp-04 | 2nd stage: 46k self-generated correct traces (RFT) + 40k unseen rows | 0.7533 (−2.7, 0.9σ) | iterate |
| exp-05 | rescore both candidates at n=500 | 0.772 vs 0.770 — tied at 0.24σ | reject |
| exp-06 | weight average of exp-02 and exp-04 | 0.782 at n=500 (best of the three) | adopt |
| exp-07 | probe: does the grader's 10-shot prefix hurt? | zero-shot 0.800 vs 10-shot 0.788 | reject retrain |
| exp-08 | verify `final_model/` under the harness, full test set | 0.7582 ± 0.0118 (n=1319) | adopt |
| exp-09 | SFT on prompts carrying the grader's 10-shot prefix | 0.7399 vs 0.7497 on 819 held-out items | reject |

**What actually moved the number:** teaching the model to stop (+58) and turning off the sampling the base
model's `generation_config` was silently applying (+11.3). Everything after that — more data, on-policy
rejection sampling, prefix matching — landed inside the noise; only the weight soup was (weakly) positive.

**Data:** nvidia/OpenMathInstruct-2 rev `469216e` (gsm8k and augmented_gsm8k rows) and self-generated traces
on GSM8K *train* questions. Every training file passed `contamination_check.py` against the test copy with
zero matches. No test item was used for training, prompting, or data selection.

**Reproduction:** cards in `memory/cards/exp-*.yaml` carry the exact argv, data provenance and eval paths;
`memory/index.md` is the one-line index.
