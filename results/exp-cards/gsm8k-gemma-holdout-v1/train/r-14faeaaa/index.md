# r-14faeaaa — extracted experiment cards

Base model: Qwen/Qwen3-4B-Base · benchmark: gsm8k · budget: 10 h, 1x H100.
All measurements are the agent's own `--limit 150` evals. "tuned" = its own vLLM server
(`--disable-cascade-attn`, 16 connections); "stock" = plain `python3 evaluate.py`.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| exp-01 | 419 | 0.65 | sft | base_model | work/sft_data_v1.jsonl | 2e-5 / 2 ep | killed | — | inconclusive | abandon_line |
| exp-02 | 441 | 0.67 | sft | base_model | work/sft_data_v1.jsonl | 2e-5 / 2 ep | killed | — | inconclusive | abandon_line |
| exp-03 | 454 | 0.68 | rft | base_model | work/sft_data_v1.jsonl (round-1 RFT, ~22K rows) | 2e-5 / 2 ep | completed | 0.853 tuned (150) | supported | adopt |
| exp-04 | 559 | 1.55 | rft | base_model | work/sft_data_v2.jsonl (phase-2 build, MetaMath x2) | 2e-5 / 2 ep | killed | — | inconclusive | abandon_line |
| exp-05 | 637 | 1.93 | rft | base_model | work/sft_data_v2.jsonl (47,116 rows) | 2e-5 / 2 ep | killed | — | inconclusive | abandon_line |
| exp-06 | 685 | 2.00 | rft | base_model | work/sft_data_v2.jsonl (47,116 rows) | 2e-5 / 2 ep | completed | 0.927 tuned, ckpt-599 (150) | supported | adopt |
| exp-07 | 798 | 3.95 | grpo | exp-06 | gsm8k train prompts (all) | 2e-6 / 250 steps | failed | — | inconclusive | abandon_line |
| exp-08 | 849 | 4.08 | grpo | exp-06 | gsm8k train prompts (all) | 2e-6 / 250 steps | completed | 0.913 tuned (150) | contradicted | reject |
| exp-09 | 911 | 4.68 | grpo | exp-06 | work/grpo_band_idx.json (hard band) | 1e-6 / 200 steps, 50 done | killed | 0.920 tuned & stock, ckpt-50 (150) | inconclusive | adopt |
| exp-10 | 936 | 4.94 | rft | base_model | work/sft_data_v3.jsonl (71,912 rows) | 2e-5 / 1 ep | completed | 0.913 tuned & stock (150) | contradicted | reject |
| exp-11 | 1005 | 6.58 | merge | exp-06 + exp-09 | — | — | completed | 0.920 tuned & stock x2 (150) | inconclusive | adopt |
| exp-12 | 1013 | 6.62 | decode-config | exp-06 | — | — | completed | 0.847 tuned (150) | contradicted | reject |
| exp-13 | 1021 | 6.65 | grpo | exp-06 | work/grpo_band_idx.json (hard band) | 1e-6 / 150 steps | completed | 0.927 tuned, ckpt-50 (150) | inconclusive | reject |
| exp-14 | 1028 | 6.66 | other (packaging) | exp-06 | — | — | completed | 0.907 stock (150) | inconclusive | reject |
| exp-15 | 1094 | 7.67 | merge | exp-06 + exp-09 + exp-13 | — | — | completed | 0.920 stock (150) | inconclusive | reject |
| exp-16 | 1110 | 7.78 | other (packaging) | exp-11 | — | — | completed | 0.927 stock, 0.920 rerun (150) | supported | adopt |
| exp-17 | 1133 | 7.83 | merge | exp-06 + exp-09 + exp-10 | — | — | completed | none in the stream | inconclusive | abandon_line |

exp-16 is the submitted artifact: `final_model` = weight average of the SFT epoch-1 checkpoint
(exp-06) and the hard-band GRPO step-50 checkpoint (exp-09), greedy decoding baked into
`generation_config.json`.
