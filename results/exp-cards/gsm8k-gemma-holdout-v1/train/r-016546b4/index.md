# r-016546b4 - reconstructed experiment cards

Base model: Qwen/Qwen3-1.7B-Base | benchmark: gsm8k | budget: 10 h, one H100.
7 launches carded. The stream ends at event [230], t=+3.55h, with the last
training run still going, so the run is only reconstructed to a third of its
budget.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 23 | 0.01 | other (package base -> final_model) | base_model | - | - | completed | accuracy 0.100, n=20 | inconclusive | reject |
| exp-02 | 52 | 0.10 | sft | base_model | MetaMathQA GSM 50k | 2e-5 / 1 | failed | - | inconclusive | iterate |
| exp-03 | 62 | 0.12 | sft | base_model | MetaMathQA GSM 50k | 2e-5 / 1 | completed | - (its copy measured on exp-04) | inconclusive | adopt |
| exp-04 | 68 | 0.65 | other (package v1 -> final_model) | exp-03 | - | - | completed | accuracy 0.140, n=50 | inconclusive | reject |
| exp-05 | 84 | 0.71 | sft | base_model | MetaMathQA GSM 100k, eval-format targets | 2e-5 / 2 (stalled at 1.71) | killed | - (its checkpoint measured on exp-06) | inconclusive | adopt |
| exp-06 | 119 | 3.39 | other (package checkpoint-3125 + chat template -> final_model) | exp-05 | - | - | completed | accuracy 0.300, n=20 (base 0.100 same limit) | supported | adopt |
| exp-07 | 201 | 3.48 | sft | base_model | MetaMathQA GSM + gsm8k train, 150k of 247,473 | 2e-5 / 3 | killed | - | inconclusive | abandon_line |

Submission: exp-06 - final_model holding trained_model_v2/checkpoint-3125 with
the chat template inlined into tokenizer_config.json. exp-03 and exp-05 are
marked adopt as parents of the packaging cards downstream of them; only exp-06's
output is the final_model the stream leaves behind.

Smoke tests: none. Every launch was meant to produce a candidate.
