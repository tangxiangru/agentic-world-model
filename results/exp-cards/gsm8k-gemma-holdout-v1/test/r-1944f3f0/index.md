# r-1944f3f0 — extracted experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h, 1x H100 80GB
The digest carries no event timestamps, so `elapsed_h` is null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 128 | null | sft | base_model | openai/gsm8k main/train (0.995 split) | 1e-4 / 2.0 | killed | none (killed at step 88/1860) | inconclusive | abandon_line |
| exp-02 | 254 | null | sft | base_model | openai/gsm8k main/train (0.995 split) | 1e-4 / 0.35 | completed | accuracy 0.300 @ n=40 (expq_eval40.json), +0.275 vs base 0.025; 0.347 @ n=150 | supported | adopt |
| exp-03 | 319 | null | sft | base_model | openai/gsm8k main/train (0.995 split) | 1e-4 / 1.0 | killed | none (killed at step 181/930) | inconclusive | abandon_line |
| exp-04 | 357 | null | decode-config | exp-02 | none | null / null | completed | accuracy 0.275 @ n=40 (expq_det_eval40.json), -0.025 vs exp-02 | contradicted | reject |
| exp-05 | 361 | null | sft | base_model | openai/gsm8k main/train (0.995 split) | 1e-4 / 0.6 | killed | none (killed at step 254/558) | inconclusive | abandon_line |
| exp-06 | 394 | null | other (packaging) | exp-02 | none | null / null | completed | accuracy 0.373 @ n=150 (final_model_eval150.json), +0.027 vs exp-02 0.347 | supported | adopt |

Submitted artifact: `final_model` — exp-06, a copy of exp-02's merged checkpoint `runs/expq_merged`.
Not cards (recorded as `provenance.smoke_runs` on exp-01): [31] aborted base-model baseline eval, [89] 0.02-epoch training smoke, [97] smoke eval that crashed on missing Gemma-3 processor files, [110] smoke eval at 0.0/10.
