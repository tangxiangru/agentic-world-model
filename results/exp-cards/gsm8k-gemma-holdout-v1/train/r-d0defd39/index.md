# r-d0defd39 — gsm8k, HuggingFaceTB/SmolLM3-3B-Base, 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 73 | 0.62 | sft (LoRA r=64) | base_model | openai/gsm8k train 7473 | 2e-4 / 3 | completed | accuracy 0.000 (n=30) | inconclusive | reject |
| exp-02 | 105 | 2.32 | sft (LoRA r=64) | base_model | openai/gsm8k train 7473 | 2e-4 / 3 | completed | accuracy 0.040 (n=50) | contradicted | reject |
| exp-03 | 120 | 3.01 | sft (LoRA r=128) | base_model | openai/gsm8k train 7473 + nvidia/OpenMathInstruct-2 5000 (MetaMathQA failed to load) | 1e-4 / 2 | completed | accuracy 0.160 (n=50, also 0.160 at n=100) | inconclusive | adopt |
| exp-04 | 138 | 4.16 | other (packaging, cp to final_model_backup and back) | exp-03 | — | — / — | completed | accuracy 0.160 (n=50) | inconclusive | reject |
| exp-05 | 152 | 4.86 | sft (LoRA r=128) | base_model | openai/gsm8k train (never loaded) | 5e-5 / 3 | failed | — | inconclusive | abandon_line |

Notes

- Comparator throughout: the base model at accuracy 0.220 (n=50) measured at [27]-[28]; it was never tee'd to a workspace file. `baseline_eval.log` despite its name holds a `./final_model_backup` eval (0.140, n=50).
- exp-03's merged weights sit in `./final_model` from [121] to the end of the stream and are the presumed submission; no later write to that directory appears.
- [49], [53], [59] and [63] are trl/transformers API crashes of the same recipe and are recorded as `provenance.smoke_runs` on exp-01, not as cards. `prepare_data.py` was written at [79] but never launched, so it has no card. The packaging attempt at [180] produced nothing (`cp: cannot copy a directory into itself`) and is noted on exp-04 instead of getting its own card.
- The digest's last event is [209] at t=+6.35h of a 10 h budget.
