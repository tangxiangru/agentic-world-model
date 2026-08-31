# Reconstructed experiment cards — r-c51755a7

base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h, 1x H100
All accuracies are the agent's own `evaluate.py` runs at the default `--limit 150`
(n=150), except the base-model reading, which was taken at `--limit 10` (n=10).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 92 | 0.22 | sft | base_model | HuggingFaceH4/Bespoke-Stratos-17k (16,610) | 2e-5 / 2 | completed | accuracy 0.180 (n=150) | inconclusive | reject |
| exp-02 | 138 | 1.41 | sft | base_model | openai/gsm8k train (7,473) | 2e-5 / 3 | completed | accuracy 0.460 (n=150) | supported | reject |
| exp-03 | 160 | 2.37 | sft | base_model | HuggingFaceH4/Bespoke-Stratos-17k (16,610) + gsm8k train as few-shot pool | 2e-5 / 2 | completed | accuracy 0.367 (n=150) | contradicted | adopt |
| exp-04 | 186 | 4.68 | decode-config | exp-03 | none | — / — | completed | accuracy 0.447 (n=150) | supported | reject |
| exp-05 | 204 | 4.88 | sft | base_model | openai/gsm8k train (7,473) | 2e-5 / 3 | completed | accuracy 0.480 (n=150) | supported | reject |
| exp-06 | 220 | 5.83 | sft | base_model | AI-MO/NuminaMath-CoT 50k subset + gsm8k train as few-shot pool | 2e-5 / 1 | completed | accuracy 0.520 (n=150) | supported | adopt |
| exp-07 | 230 | 8.07 | other (packaging) | exp-06 | none | — / — | completed | none (not re-evaluated) | inconclusive | adopt |

Notes

- exp-03's `decision: adopt` is mechanical: its checkpoint is the one exp-04
  patched in place. Its accuracy was below exp-02 and the data line was dropped.
- exp-07 is the packaging step that put the exp-06 weights at `final_model`; it
  is the submitted card and carries `outcome.official_accuracy: null`.
- 14 aborted starts and API-error crashes are recorded as
  `provenance.smoke_runs` on exp-01, exp-03, exp-05 and exp-06 rather than as
  cards of their own.
