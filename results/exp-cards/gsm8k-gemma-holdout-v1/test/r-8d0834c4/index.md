# r-8d0834c4 — reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 51 | 0.25 | sft | base_model | gsm8k_train.json (7473) | 2e-4 / 3 capped at max_steps=1000 | completed | none (adapter not evaluated) | inconclusive | adopt |
| exp-02 | 59 | 0.49 | merge | exp-01 | — | — | completed | accuracy 0.200 (n=20, +0.150 vs base_model 0.050); 0.200 (n=50) | supported | reject |
| exp-03 | 86 | 0.70 | sft | base_model | gsm8k_train.json (7473) | 2e-4 / 3 capped at max_steps=2000 (epoch 2.14) | completed | none (adapter not evaluated) | inconclusive | adopt |
| exp-04 | 88 | 1.16 | merge | exp-03 | — | — | completed | accuracy 0.300 (n=50, +0.100 vs exp-02 0.200) | supported | reject |
| exp-05 | 96 | 1.23 | sft | base_model | gsm8k_train.json (7473) | 2e-4 / 3 exceeded at max_steps=3000 (epoch 3.21) | completed | none (adapter not evaluated) | inconclusive | adopt |
| exp-06 | 98 | 1.95 | merge | exp-05 | — | — | completed | accuracy 0.260 (n=50, -0.040 vs exp-04 0.300); 0.280 (n=100) | contradicted | adopt |

Notes
- exp-06 is the submitted artifact: its merge is the last write to `/home/ben/task/final_model` and the run ends with that directory intact.
- Four launches that died before a training step ([29], [35], [41], [47] — JSON-lines parse, then three SFTTrainer kwarg rejections) are recorded as `provenance.smoke_runs` on exp-01, not as cards.
- The agent stated no problem and no hypothesis before any launch, so every card carries `problem.statement: null`, `hypothesis.claim: null` and `stated_by_agent: false`.
