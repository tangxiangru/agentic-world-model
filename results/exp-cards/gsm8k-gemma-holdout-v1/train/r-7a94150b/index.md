# r-7a94150b — extracted experiment cards

Base model: Qwen/Qwen3-4B-Base · benchmark: gsm8k · budget: 10 h, 1x H100 80GB.
The digest carries no event timestamps ("no timestamps in this run"), so `elapsed_h` is null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 84 | null | sft | base_model | openai/gsm8k train (7248) | 2e-4 / 0.25 | failed | — (crashed before training) | inconclusive | iterate |
| exp-02 | 101 | null | sft | base_model | openai/gsm8k train (7248), chat format | 2e-4 / 0.25 | completed | 0.175 @ n=40 (sanity_eval40.json) | inconclusive | adopt* |
| exp-03 | 131 | null | decode-config | exp-02 | — | — | completed | 0.15 @ n=20 (sanity_eval20_1200.json), −0.35 vs base 0.50 | contradicted | reject |
| exp-04 | 159 | null | sft | base_model | openai/gsm8k train (7248), plain format | 5e-6 / 1.0 | completed | 0.25 @ n=20 (conservative_eval20_1200.json), −0.25 vs base 0.50 | contradicted | reject |
| exp-05 | 177 | null | sft | base_model | openai/gsm8k train (7248), chat format | 1e-6 / 0.03 | completed | 0.425 @ n=80 (micro_eval80_1200.json), −0.075 vs base 0.50 | contradicted | reject |
| exp-06 | 204 | null | other (packaging) | base_model | — | — | completed | 0.500 @ n=20 (final_model_eval20.json), 0.0 vs base 0.50 | supported | adopt |

\* exp-02 is `adopt` only under the rule "the output became the parent of a later card" (exp-03 operates on `runs/sanity_merged`). That checkpoint was dropped at [153] and never reached `final_model`. The submitted artefact is exp-06.

Run-level notes:
- No fine-tune ever beat the untouched base model (0.15 / 0.25 / 0.425–0.45 against 0.50), and `final_model` was filled with an unmodified copy of the base-model snapshot, so the shipped weights carry no post-training.
- Both baseline evals launched at the start ([27], [73]) were killed before producing a number; `baseline_40.json` and `baseline_20_fast.json` are not in the workspace.
- All eval numbers are the agent's own `evaluate.py` outputs in the workspace; several eval console blocks are absent from the digest and were read from the JSON files.
