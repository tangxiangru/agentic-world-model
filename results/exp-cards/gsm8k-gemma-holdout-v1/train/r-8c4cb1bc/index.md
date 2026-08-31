# r-8c4cb1bc — experiment cards (extracted)

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 159 | null | sft | base_model | openai/gsm8k train (7217) | 2e-4 / 3 | completed | held-out loss 0.4556 (epoch 1, n=256) | inconclusive | adopt |
| exp-02 | 202 | null | merge | exp-01 | — | null / null | completed | accuracy 0.387 (n=150) | inconclusive | adopt |
| exp-03 | 213 | null | sft | base_model | openai/gsm8k x6 + meta-math/MetaMathQA GSM_Rephrased,GSM_AnsAug (<=60000) | 1.5e-4 / 1 | killed | held-out loss 0.5005 (step 400, n=256) | inconclusive | abandon_line |
| exp-04 | 281 | null | sft | base_model | openai/gsm8k train | 2e-4 / 1 | completed | held-out loss 0.4112 (n=256) | inconclusive | adopt |
| exp-05 | 292 | null | merge | exp-04 | — | null / null | completed | accuracy 0.087 (n=150) | contradicted | reject |
| exp-06 | 317 | null | sft | base_model | openai/gsm8k train | 2e-4 / 5 | completed | accuracy 0.08 (epoch 3, n=50) | contradicted | reject |
| exp-07 | 411 | null | merge | exp-01 | — | null / null | completed | accuracy 0.140 (n=50) | contradicted | reject |
| exp-08 | 514 | null | sft | base_model | openai/gsm8k train, evaluator-aligned 10-shot format | 2e-4 / 2 | completed | accuracy 0.18 (checkpoint-452, n=50) | contradicted | reject |
| exp-09 | 752 | null | merge | exp-01 | — | null / null | completed | accuracy 0.127 (n=150) | contradicted | reject |
| exp-10 | 793 | null | other (packaging) | exp-02 | — | null / null | completed | accuracy 0.3867 (n=150) | supported | adopt |

Smoke runs (not cards, recorded on exp-01): [141] crashed — LoRA + gradient checkpointing, inputs not marked to require grads; [151] passed (1 step).

The shipped artifact is exp-10: `final_model`, a copy of `runs/pilot_gsm8k_merged` (the merged epoch-3 adapter of the three-epoch GSM8K-only LoRA run, exp-01/exp-02).
