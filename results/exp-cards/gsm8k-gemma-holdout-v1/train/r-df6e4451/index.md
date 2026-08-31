# Reconstructed experiment cards — run r-df6e4451

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h on one H100.
Ten launches, in launch order. `elapsed_h` is the `t=` on the launch event.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 427 | 0.47 | sft | base_model | train_v1.jsonl (gsm8k human x3 + OpenMathInstruct-2 gsm8k-sourced 200K) | 1e-5 / 1 | killed | none | inconclusive | abandon_line |
| exp-02 | 510 | 0.54 | sft | base_model | train_v1.jsonl | 1e-5 / 1 | killed | none | inconclusive | abandon_line |
| exp-03 | 588 | 0.60 | sft | base_model | train_v1.jsonl | 1e-5 / 1 | failed | none (CUDA OOM in backward) | inconclusive | abandon_line |
| exp-04 | 697 | 0.70 | sft | base_model | train_v1.jsonl | 1e-5 / 1 | killed | none | inconclusive | abandon_line |
| exp-05 | 714 | 0.72 | sft | base_model | train_v1.jsonl (220658 samples) | 1e-5 / 1 | completed | accuracy 0.793 @ --limit 150 (dev250 0.844) | inconclusive | adopt |
| exp-06 | 1106 | 3.29 | sft | base_model | train_v2.jsonl (v1 mix, OMI cut to 115K, + RFT traces from exp-05 x2) | 1e-5 / 1 | completed | accuracy 0.753 @ --limit 150 (dev250 0.816) | contradicted | reject |
| exp-07 | 1167 | 5.36 | merge | exp-05 (+ exp-06) | none | - / - | completed | accuracy 0.836 @ dev250 | inconclusive | reject |
| exp-08 | 1247 | 5.56 | grpo | exp-05 | gsm8k train prompts minus dev500, verifiable answer reward | 2e-6 / - (200 steps requested) | killed | none | inconclusive | abandon_line |
| exp-09 | 1271 | 6.34 | grpo | exp-05 | gsm8k train prompts minus dev500, verifiable answer reward | 2e-6 / - (100 of 150 steps) | killed | accuracy 0.896 @ dev250 (+0.052 vs exp-05) | supported | adopt |
| exp-10 | 1348 | 8.16 | decode-config | exp-09 | none (packaging to final_model) | - / - | completed | accuracy 0.840 @ --limit 150 (rerun 0.8467) | supported | adopt |

Submitted artifact: exp-10 (`/home/ben/task/final_model`, the exp-09 GRPO checkpoint-100 exported with greedy `generation_config.json` and eos `[128001, 128012]`).

Not cards (smoke / dry runs, recorded as `provenance.smoke_runs`): [257], [362], [402] on exp-01; [739] on exp-06; [1171], [1205] on exp-08. The base-model baseline eval at [136] is not a launch; it is the comparator on exp-01 to exp-05 (0.24 at `--limit 50`).
