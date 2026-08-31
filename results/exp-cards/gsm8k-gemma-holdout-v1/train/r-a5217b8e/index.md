# Experiment cards - r-a5217b8e (Qwen/Qwen3-4B-Base, gsm8k, 10 h, 1x H100)


30 cards, one per launch that produced a candidate. `elapsed_h` is null throughout: the digest's block
headers carry no timestamps for this run. Measurements are the agent's own evals; `official_accuracy` is
never written. The submission is exp-21.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | [300] | null | merge | base_model | adapter runs/exp1_gsm_only_checkpoint468_keep | - | completed | none | inconclusive | abandon_line |
| exp-02 | [301] | null | merge | base_model | adapter runs/exp1_gsm_only | - | completed | 0.10 @50 | inconclusive | reject |
| exp-03 | [328] | null | decode-config | exp-02 | none (stop-token patch) | - | completed | 0.15 @20 | inconclusive | reject |
| exp-04 | [339] | null | merge | base_model | adapter runs/exp1_gsm_only | - | completed | none (eval killed) | inconclusive | abandon_line |
| exp-05 | [358] | null | sft | base_model | gsm8k train (plain fmt) | 1e-4 / 2 ep | completed | 0.60 @50 | inconclusive | adopt |
| exp-06 | [421] | null | sft | exp-05 | gsm8k x2 + TemplateGSM 15k | 5e-5 / 1 ep | completed | 0.35 @20 | contradicted | reject |
| exp-07 | [503] | null | sft | base_model | gsm8k train[:128] | 5e-5 / 0.05 ep | completed | 0.44 @50 | contradicted | reject |
| exp-08 | [531] | null | merge | exp-05 | adapter checkpoint-234 | - | completed | 0.56 @50 | contradicted | reject |
| exp-09 | [545] | null | other (packaging) | exp-05 | - | - | completed | 0.68 @100 | inconclusive | adopt |
| exp-10 | [614] | null | sft | exp-05 | gsm8k train (rendered_chat) | 5e-5 / 0.25 ep | completed | 0.73 @100 | supported | adopt |
| exp-11 | [641] | null | sft | exp-10 | gsm8k train (rendered_chat) | 3e-5 / 0.5 ep | completed | 0.76 @100 | supported | adopt |
| exp-12 | [660] | null | sft | exp-11 | gsm8k train (rendered_chat) | 2e-5 / 0.5 ep | completed | 0.74 @100 | contradicted | reject |
| exp-13 | [674] | null | sft | exp-11 | gsm8k train (rendered_chat) | 1e-5 / 0.25 ep | completed | 0.78 @100 | supported | adopt |
| exp-14 | [687] | null | sft | exp-13 | gsm8k train (rendered_chat) | 5e-6 / 0.25 ep | completed | 0.75 @100 | contradicted | reject |
| exp-15 | [701] | null | other (packaging) | exp-13 | - | - | completed | 0.78 @150 | inconclusive | adopt |
| exp-16 | [823] | null | distill | exp-13 | teacher_full.jsonl (7143) | 5e-6 / 0.25 ep | completed | 0.853 @150 | supported | adopt |
| exp-17 | [897] | null | other (packaging) | exp-16 | - | - | completed | 0.8533 @150 (copied) | inconclusive | adopt |
| exp-18 | [901] | null | sft | exp-16 | gsm8k train (rendered_chat) | 3e-6 / 0.0625 ep | completed | 0.76 @100 | contradicted | reject |
| exp-19 | [922] | null | distill | exp-13 | teacher_full.jsonl (7143) | 5e-6 / 0.125 ep | completed | 0.76 @100 | contradicted | reject |
| exp-20 | [1026] | null | distill | exp-17 | teacher_full + self-repairs x4 | 3e-6 / 0.125 ep | completed | 0.87 @100 | supported | adopt |
| exp-21 | [1064] | null | other (packaging) | exp-20 | - | - | completed | 0.87 @100 / 0.8667 @150 (copied) | inconclusive | adopt |
| exp-22 | [1210] | null | distill | exp-21 | teacher_full.jsonl + 10-shot sys prompt | 2e-6 / 0.0625 ep | completed | 0.87 @100 | inconclusive | adopt |
| exp-23 | [1234] | null | distill | exp-22 | teacher_full + full self-distill | 2e-6 / 0.031 ep | completed | 0.81 @100 | contradicted | reject |
| exp-24 | [1248] | null | distill | exp-21 | teacher_full + 221 new misses x8 | 2e-6 / 0.125 ep | completed | 0.84 @100 | contradicted | reject |
| exp-25 | [1314] | null | decode-config | exp-21 | none (EOS/stop change) | - | completed | not reported | inconclusive | reject |
| exp-26 | [1341] | null | distill | exp-21 | teacher_full + 308 own misses x6 | 2e-6 / 0.125 ep | completed | 0.79 @100 | contradicted | reject |
| exp-27 | [1488] | null | distill | exp-21 | teacher_full_rationalized_fewshot (7333 kept) | 2e-6 / 0.0625 ep | completed | 0.65 @100 | contradicted | reject |
| exp-28 | [1505] | null | distill | exp-21 | teacher_full_rationalized_fewshot | 2e-6 / 0.0625 ep | completed | 0.81 @100 | contradicted | reject |
| exp-29 | [1519] | null | distill | exp-21 | teacher_full + self-repairs x4 (few-shot) | 1e-6 / 0.031 ep | completed | 0.82 @100 | contradicted | reject |
| exp-30 | [1616] | null | dpo | exp-21 | dpo_pairs_full_exact.jsonl (537) | 5e-7 / 1.0 ep | completed | 0.83 @100 | contradicted | reject |
