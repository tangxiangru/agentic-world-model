# Reconstructed experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base. Benchmark: gsm8k. Budget: 10 h, 1x H100.
Workspace root in-run: `/home/ben/task`. 8 cards; `exp-08` is the exported candidate.
Accuracies are the agent's own `evaluate.py` runs; `official_accuracy` is not written on any card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 89 | 0.23 | sft | base_model | MetaMathQA GSM_* (240,000) | 2e-5 / 1 | failed | none (CUDA OOM at step 100/1875) | inconclusive | iterate |
| exp-02 | 117 | 0.42 | sft | base_model | MetaMathQA GSM_* (240,000) | 2e-5 / 1 | completed | 0.7067 @150 (eval_stage1_full150.json) | inconclusive | adopt |
| exp-03 | 187 | 2.12 | sft | exp-02 | openai/gsm8k train (7,473), evaluator ten-shot prompt | 5e-6 / 1 | completed | 0.4933 @150 (eval_stage2_full150.json) | contradicted | reject |
| exp-04 | 242 | 2.63 | sft | exp-02 | MetaMathQA GSM_* (240,000), 2nd pass | 1e-5 / 1 | completed | 0.72 @150 (eval_epoch2_full150.json) | inconclusive | adopt |
| exp-05 | 289 | 4.42 | sft | exp-04 | MetaMathQA GSM_* (240,000), 3rd pass | 5e-6 / 1 | completed | 0.7067 @150 (eval_epoch3_half150.json) | inconclusive | reject |
| exp-06 | 486 | 6.12 | sft | exp-04 | MetaMathQA MATH_* (155,000) | 3e-6 / 1 | completed | 0.6667 @150 (eval_math_half150.json) | contradicted | reject |
| exp-07 | 510 | 7.32 | sft | exp-04 | MetaMathQA GSM_AnsAug+GSM_Rephrased (160,000) | 2e-6 / 1 | completed | 0.82 @150 greedy (eval_forward_full_greedy150.json); 0.7597 @1319 (eval_forward_full_greedy_fulltest.json) | supported | adopt |
| exp-08 | 594 | 8.74 | other (packaging) | exp-07 | none | n/a | completed | 0.82 @150 evaluator defaults (final_eval_default.json) | supported | adopt |

## Not written as cards

- Two pipeline smoke runs before the first real training launch ([83], [86], `--limit 512`,
  `--skip-final-save`) and one before the exact-prompt launch ([182], `--limit 128`); recorded as
  `provenance.smoke_runs` on exp-01 and exp-03.
- A checkpoint interpolation, `interp_exact_010`, evaluated at 0.687 on the 150-item slice
  ([238], `eval_interp010_150.json`). `interpolate_models.py` is in the workspace but the
  invocation that produced the merged directory never appears in the stream, so there is no
  launch to cite. Noted in exp-03 `provenance.unresolved`.
- The decode change that pinned the candidate's generation defaults to `do_sample=false` /
  `temperature=0.0`, announced at [558] and confirmed at [598]/[612]. It is worth roughly
  +10 points on the 150-item slice, but the command or edit that wrote the config is absent from
  the stream, so its measurements are recorded on exp-07 and the gap is noted in exp-07
  `provenance.unresolved`.
