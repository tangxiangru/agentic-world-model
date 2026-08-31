# r-f4df2c91 — reconstructed experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h, 1x H100.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 20 | null | other (package base -> final_model) | base_model | none | null | completed | 0.09 @ n=100 [24] | inconclusive | reject |
| exp-02 | 81 | null | merge (adapter in final_model) | base_model | none | null | completed | none stated | inconclusive | reject |
| exp-03 | 96 | null | merge (./checkpoints) | base_model | none | null | failed | none | inconclusive | abandon_line |
| exp-04 | 102 | null | merge (checkpoints/checkpoint-126) | base_model | none | null | completed | 0.04 @ n=100 [172] | contradicted | reject |
| exp-05 | 128 | null | merge (./checkpoints_quick) | base_model | none | null | failed | none | inconclusive | abandon_line |
| exp-06 | 130 | null | merge (checkpoints_quick/checkpoint-64) | base_model | none | null | completed | 0.06 @ n=100 [172] | contradicted | reject |
| exp-07 | 144 | null | other (restore base -> final_model) | base_model | none | null | completed | 0.12 @ n=100 [159] | inconclusive | reject |
| exp-08 | 153 | null | merge (checkpoints_final/checkpoint-117) | base_model | none | null | completed | 0.03 @ n=100 [159] | contradicted | reject |
| exp-09 | 160 | null | other (package base -> final_model, submitted) | base_model | none | null | completed | 0.127 @ n=150 [164] | inconclusive | adopt |

Run-level notes:

- The three trainings that actually ran (the 126-step run over 2000 examples,
  the 64-step run over 1000 examples, and the 117-step full-set run, plus a
  full-set run killed at step 342 of 702) have **no cards**: their launch
  commands do not appear anywhere in the digest, only their kills
  (`kill -9 323197` [67], `pkill -f "python train_v2.py"` [122]), their merges,
  and their logs in the workspace. Every card here is a merge or a packaging
  step, which are the only candidate-producing launches that can be pointed at.
- The agent never wrote an eval output file; there is no `eval_*.json` or
  `baseline_results.json` in the workspace, so every measurement is the agent's
  own spoken number with `path: null`.
- No timestamps in the digest, so `elapsed_h` and `wall_h` are null throughout.
