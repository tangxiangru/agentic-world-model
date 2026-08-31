# r-4463b5d3 — extracted experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, one H100.
15 launches, all cited by event index. `best measurement` is the agent's own eval of that
card's output; `—` means the launch produced no measurement of its own.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 402 | 0.15 | sft | base_model | sft_v1.jsonl (--limit 4000) | 1e-5 / 1 | killed | — | inconclusive | abandon_line |
| exp-02 | 451 | 0.17 | sft | base_model | sft_v1.jsonl (--limit 6000) | 1e-5 / 1 | killed | — | inconclusive | abandon_line |
| exp-03 | 470 | 0.25 | sft | base_model | sft_v1.jsonl (--limit 6000) | 1e-5 / 1 | killed | — | inconclusive | abandon_line |
| exp-04 | 483 | 0.32 | sft | base_model | sft_v1.jsonl (--limit 6000) | 1e-5 / 1 | killed | — | inconclusive | abandon_line |
| exp-05 | 532 | 0.33 | sft | base_model | sft_v1.jsonl (117,650) | 1.4e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-06 | 632 | 0.39 | sft | base_model | sft_v1.jsonl (117,650) | 1.4e-5 / 2 | completed | 0.695 @ n=200 | inconclusive | adopt |
| exp-07 | 896 | 3.21 | other (packaging) | exp-06 | — | — | completed | — (copy of exp-06) | inconclusive | reject |
| exp-08 | 974 | 3.25 | grpo | exp-06 | rl_pool_full.jsonl (96 prompts) | 1e-6 / 1 | killed | — | inconclusive | abandon_line |
| exp-09 | 998 | 3.26 | grpo | exp-06 | rl_pool_full.jsonl (96 prompts) | 1e-6 / 1 | completed | — | inconclusive | abandon_line |
| exp-10 | 1040 | 3.32 | grpo | exp-06 | rl_pool_full.jsonl (20,000 prompts) | 1.5e-6 / 1 | failed (OOM) | — | inconclusive | abandon_line |
| exp-11 | 1079 | 3.35 | grpo | exp-06 | rl_pool_full.jsonl (20,000 prompts) | 1.5e-6 / 1 | killed | 0.866 @ n=800 (0.850 @ n=200) | inconclusive | adopt |
| exp-12 | 1245 | 6.31 | other (packaging) | exp-11 | — | — | completed | — (copy of exp-11) | inconclusive | reject |
| exp-13 | 1247 | 6.31 | grpo | exp-11 | rl_pool_full.jsonl (12,000 prompts, --skip 12000) | 1.5e-6 / 1 | killed | 0.874 @ n=800 (0.890 @ n=300) | inconclusive | adopt |
| exp-14 | 1325 | 8.25 | other (packaging) | exp-13 | — | — | completed | — (copy of exp-13) | inconclusive | adopt |
| exp-15 | 1343 | 8.25 | other (bf16 re-save) | exp-14 | — | — | completed | 0.8696 @ n=1319 | inconclusive | adopt |

Notes
- exp-15's output is the submission: ckpt/grpo_v2/checkpoint-50 re-saved in bf16 as final_model.
- Every verdict is `inconclusive`: the agent never stated a hypothesis before a launch, and the
  two largest jumps are measured across different `--limit` values (baseline n=150 → SFT n=200 →
  GRPO n=200/800 → final n=1319).
- Data-prep runs (prep_data.py, make_dev.py, make_rft_pool.py), the rejection-sampling
  generation at [817] (its output was never trained on) and the eval-only launches are not
  cards; they are cited from the cards they belong to.
