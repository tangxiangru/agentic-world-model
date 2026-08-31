# r-0788f765 — extracted experiment cards

base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 84 | 0.26 | sft | base_model | meta-math/MetaMathQA (395k, seed-42 shuffle) | 2e-5 / 1 | failed | — | inconclusive | iterate |
| exp-02 | 96 | 0.45 | sft | base_model | meta-math/MetaMathQA (395k, seed-42 shuffle) | 2e-5 / 1 | failed | — | inconclusive | iterate |
| exp-03 | 104 | 0.63 | sft | base_model | meta-math/MetaMathQA (395k, seed-42 shuffle) | 2e-5 / 1 | failed | — | inconclusive | iterate |
| exp-04 | 112 | 0.81 | sft | base_model | meta-math/MetaMathQA (395k, seed-42 shuffle) | 2e-5 / 1 | failed | — | inconclusive | iterate |
| exp-05 | 122 | 1.00 | sft | base_model | meta-math/MetaMathQA (395k, seed-42 shuffle) | 2e-5 / 1 | completed | accuracy 0.100, n=10 (metrics.json) | inconclusive | adopt |
| exp-06 | 148 | 6.23 | decode-config | exp-05 | — | — / — | completed | accuracy 0.500, n=10 (test2.json) | supported | adopt |

Notes: exp-01 through exp-04 are four launches of the same script that never produced a
checkpoint (one silent death, three identical OOMs at step 108/12344); the eight pipeline
smoke runs that preceded them are recorded on exp-01 as `provenance.smoke_runs`. exp-05 is
the only completed training run and its checkpoint is the parent of exp-06, the eos-token
config change that stands in `final_model` when the run ends at t=+6.26h.
