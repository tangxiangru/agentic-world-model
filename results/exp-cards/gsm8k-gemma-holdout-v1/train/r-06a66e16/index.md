# r-06a66e16 — gsm8k / Qwen3-1.7B-Base / 10 h / 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 238 | 0.48 | sft | base_model | data/sft_mix.jsonl (~284k: OMI2-gsm, Orca-Math, MetaMathQA-GSM, OMI2-math, GSM8K train x2) | 2e-5 / 1 ep | failed | none (OOM at step 0/1479) | inconclusive | abandon_line |
| exp-02 | 278 | 0.54 | sft | base_model | data/sft_mix.jsonl (same build) | 2e-5 / 1 ep | killed | none (superseded at step 18/1479) | inconclusive | abandon_line |
| exp-03 | 337 | 0.68 | sft | base_model | data/sft_mix.jsonl (~284k, comma-grouped answers, ~192M tok) | 2e-5 / 1 ep | completed | 0.807 @ limit 150 (results_sft1_greedy150.json) | inconclusive | adopt |
| exp-04 | 441 | 3.48 | grpo | exp-03 | data/gsm8k train prompts (~7.46k), scorer-faithful binary reward | 1e-6 / 180 steps | completed | 0.853 @ 150, 0.852 @ 500 (results_qwen3_grpo1_pkg_lim150_t0.0.json, results_final180_lim500.json) | supported | adopt |
| exp-05 | 576 | 5.74 | grpo | exp-04 | data/gsm8k train prompts (same set) | 1e-6 / 150 steps | completed | 0.878 @ 500, 0.860 @ 150 (results_grpo2_lim500.json) | supported | adopt |
| exp-06 | 677 | 7.62 | decode-config | exp-05 | — (packaging into final_model) | — | completed | 0.873 @ 150 (results_final_model_150.json) | inconclusive | adopt |
| exp-07 | 699 | 7.84 | grpo | exp-05 | data/train_pass_rates.jsonl, 1,721 prompts with pass<=5/6 | 1e-6 / 75 steps | completed | 0.884 @ 500, 0.860 @ 150 (results_grpo3_lim500.json) | inconclusive | reject |
| exp-08 | 800 | 8.91 | decode-config | exp-07 | — (packaging into final_model) | — | completed | 0.853 @ 150 (results_final_model_v2_150.json) | contradicted | reject |
| exp-09 | 810 | 8.95 | decode-config | exp-05 | — (packaging into final_model) | — | completed | 0.873 @ 150 (results_final_model_FINAL_150.json) | supported | adopt |

Submitted artifact: exp-09 — final_model repackaged from the GRPO-2 checkpoint (exp-05 weights),
0.873 at --limit 150 and 0.878 at n=500, against a base-model baseline of 0.110 at --limit 100.
