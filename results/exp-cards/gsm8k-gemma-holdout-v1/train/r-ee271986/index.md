| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 129 | 0.20 | sft | base_model | data/sft_v1.jsonl (47,473: gsm8k train + MetaMathQA GSM) | 1e-5 / 3 ep | completed | 0.840 acc @150 (checkpoint-2968) | inconclusive | adopt |
| exp-02 | 275 | 1.84 | decode-config | exp-01 | - | - / - | completed | 0.840 acc @150 (same checkpoints, greedy) | inconclusive | adopt |
| exp-03 | 309 | 1.99 | other (package to final_model) | exp-01 | - | - / - | completed | none | inconclusive | adopt |
| exp-04 | 342 | 2.13 | rft | base_model | data/sft_v2.jsonl (73,279: sft_v1 + self-sampled) | 1e-5 / 3 ep | failed | none (crashed at step 2290) | inconclusive | abandon_line |
| exp-05 | 422 | 3.17 | rft | base_model | data/sft_v2.jsonl (73,279: sft_v1 + self-sampled) | 1e-5 / 2 ep | completed | 0.820 acc @150 | contradicted | reject |
| exp-06 | 500 | 4.97 | grpo | exp-01 | openai/gsm8k train (in-process) | 1e-5 / 250 steps | completed | none (adapter only) | inconclusive | adopt |
| exp-07 | 528 | 5.31 | merge | exp-01 | ckpts/grpo adapter | - / - | failed | none | inconclusive | abandon_line |
| exp-08 | 542 | 5.34 | merge | exp-01 | ckpts/grpo adapter | - / - | completed | 0.8635 acc @1319 | inconclusive | adopt |
| exp-09 | 580 | 5.93 | other (package to final_model) | exp-08 | - | - / - | completed | 0.840 acc @300; 0.820 @150 grader defaults | inconclusive | adopt |
| exp-10 | 595 | 5.97 | grpo | exp-08 | openai/gsm8k train (in-process) | 1e-5 / 300 steps | completed | none (adapter only) | inconclusive | adopt |
| exp-11 | 621 | 6.35 | merge | exp-08 | ckpts/grpo2 adapter | - / - | completed | none (eval crashed, EngineDeadError) | inconclusive | abandon_line |
| exp-12 | 801 | 7.43 | grpo | exp-01 | openai/gsm8k train (in-process) | 1.5e-5 / 450 steps | completed | none (adapter only) | inconclusive | adopt |
| exp-13 | 829 | 7.99 | merge | exp-01 | ckpts/grpo3 adapter | - / - | completed | none (eval crashed, 500s) | inconclusive | abandon_line |

Submitted deliverable: exp-09 (final_model = the merged GRPO model produced by exp-08).
