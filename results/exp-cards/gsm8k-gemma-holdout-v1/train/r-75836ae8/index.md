# r-75836ae8 — reconstructed experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h, 1x H100.
The digest carries no event timestamps, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 124 | null | sft | base_model | data/gsm_only (7,473; gsm8k train only) | 1e-5 / 3 | completed | 0.01 @100 (exp1_gsm100.json); 0.00 @20 after the stop-token patch (exp1_fix20.json) | inconclusive | reject |
| exp-02 | 194 | null | sft | base_model | data/gsm_only_evalstyle (7,473; + 10-shot evaluator system message, seed 42) | 8e-6 / 2 | completed | 0.10 @20 for both checkpoint-468 and merged (exp2_ckpt468_20.json, exp2_merged_20.json) | contradicted | adopt |
| exp-03 | 348 | null | sft | exp-02 | data/gsm_only_evalstyle_eos (7,473; + `<\|endoftext\|>` suffix) | 2e-6 / 0.5 | completed | 0.35 @20 (exp3_20.json), 0.34 @100 (exp3_100.json); same weights with restored non-greedy decoding: 0.50 @100 (exp3_sampled_100.json), 0.4936 @1319 (exp3_sampled_full.json) | supported | adopt |
| exp-04 | 463 | null | sft | exp-03 | data/gsm_only_evalstyle_think_eos (n unknown; + thinking tags) | 1e-6 / 0.75 | completed | none — never benchmarked; greedy probe showed collapse into repeated thinking tokens [562] | inconclusive | abandon_line |
| exp-05 | 597 | null | other (packaging + decode config) | exp-03 | — | — | completed | 0.53 @100 with seed 42 pinned (final_model_100_seed42.json); 0.45 @100 unseeded (final_model_100.json) | supported | adopt |

Submitted artifact: **exp-05** — `final_model`, a copy of exp-03's merged checkpoint with the
restored non-greedy generation config and `seed: 42` pinned.

Not cards: the LoRA smoke run at [82] (recorded on exp-01 as `provenance.smoke_runs`), and the
data-prep invocations at [73], [93], [94], [95], [188], [342], [454] (recorded as
`setup.data[].build_command`). The two generation-config edits that changed measured accuracy —
restoring non-greedy sampling defaults on exp-03's checkpoint [522] and pinning the seed on
final_model [615] — have no command in the digest and so could not be given their own cards;
they are recorded on exp-03 and exp-05 respectively, with notes in `provenance.unresolved`.
