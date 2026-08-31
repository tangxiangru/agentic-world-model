# r-8e27ac7b — reconstructed experiment cards

Base model: `google/gemma-3-4b-pt` · benchmark: gsm8k · budget: 10 h, 1x H100.
`elapsed_h` is null on every card: the digest header states this run carries no
event timestamps.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 81 | null | sft | base_model | openai/gsm8k train (7100) | 2e-4 / 0.75 | failed | — | inconclusive | iterate |
| exp-02 | 94 | null | sft | base_model | openai/gsm8k train (7100) | 2e-4 / 0.75 | completed | 0.15 @ n=20 (metrics_pilot120_20.json) | supported | adopt |
| exp-03 | 136 | null | merge | exp-02 | — | — / — | completed | — | inconclusive | iterate |
| exp-04 | 139 | null | merge | exp-02 | — | — / — | completed | — | inconclusive | iterate |
| exp-05 | 175 | null | merge | exp-02 | — | — / — | completed | 0.19 @ n=100 (metrics_pilot120_100.json) | supported | reject |
| exp-06 | 196 | null | sft | base_model | openai/gsm8k train (7100) | 8e-5 / 1.0 | killed | 0.30 @ n=100 (metrics_run2_80_100.json) | supported | adopt |
| exp-07 | 229 | null | merge | exp-06 | — | — / — | completed | 0.30 @ n=100 (metrics_run2_80_100.json) | supported | adopt |
| exp-08 | 264 | null | other (packaging) | exp-07 | — | — / — | completed | 0.28 @ n=50 (metrics_final_50.json) | inconclusive | adopt |

Submitted model: `final_model`, packaged by **exp-08** as a copy of
`outputs/merged_run2_80` (exp-07, the merge of exp-06's step-80 LoRA adapter).

Run-level notes:

- Comparator chain: base model 0.05 @ n=20 (`metrics_base_20.json`) → pilot
  export 0.15 @ n=20 / 0.19 @ n=100 → run2 step-80 export 0.25 @ n=20 / 0.30 @
  n=100 → final_model 0.28 @ n=50.
- Two evals of the base model were launched and killed before finishing: the
  first baseline at `[46]` (`--limit 40`, killed `[62]`) and the 100-sample
  control at `[247]` (killed `[257]`), so there is no base comparator at n=100.
  Neither is a card: neither produced a candidate.
- The digest blocks `[57]`, `[78]` and `[157]` carry `ps` output from unrelated
  jobs sharing the host; none of it belongs to this run.
- The run finished with 8:48 of the 10 h budget unspent (`[274]`).
