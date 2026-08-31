# Reconstructed experiment cards - r-6c4c222b

Base model post-trained: `Qwen/Qwen3-1.7B-Base`. Benchmark: gsm8k. Budget: 10 h, one H100.
14 cards, in launch order. Smoke tests and dry runs are not cards; they are listed under
`provenance.smoke_runs` on the next real launch's card.

| exp-NN | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 189 | 0.29 | sft | base_model | data/train.jsonl (MetaMathQA GSM + gsm8k train, 96846) | 1e-5 / 3 | failed | none (OOM before any save) | inconclusive | abandon_line |
| exp-02 | 200 | 0.33 | sft | base_model | data/train.jsonl (96846) | 1e-5 / 3 | killed | none (killed at step 268) | inconclusive | abandon_line |
| exp-03 | 270 | 0.46 | sft | base_model | data/train.jsonl (96846) | 1e-5 / 3 | completed | 0.220 @ n=200 (metrics_sft_ep3.json) | inconclusive | reject |
| exp-04 | 476 | 2.30 | sft | base_model | data/train_fs.jsonl (few-shot, 20000 of 60000) | 1.8e-5 / 2 | killed | none (killed mid-run) | inconclusive | abandon_line |
| exp-05 | 510 | 2.45 | sft | base_model | data/train_fs.jsonl (15000 of 60000) | 1.5e-5 / 2 | completed | 0.695 @ n=200 (metrics_val_eot.json) | supported | adopt |
| exp-06 | 553 | 2.99 | sft | base_model | data/train_fs.jsonl (40000 of 60000) | 1.5e-5 / 2 | completed | 0.670 @ n=200 (metrics_main_ep2.json) | inconclusive | adopt |
| exp-07 | 641 | 4.40 | grpo | exp-06 | gsm8k train questions (in-process) | 2e-6 / 400 steps | completed | 0.782 @ n=500 (metrics_grpo_500.json) | supported | adopt |
| exp-08 | 675 | 4.90 | grpo | exp-07 | gsm8k train questions | 2e-6 / 400 steps | completed | 0.775 @ n=200 (metrics_grpo2.json) | inconclusive | reject |
| exp-09 | 703 | 5.38 | grpo | exp-07 | gsm8k train questions | 3e-6 / 300 steps (16 gens) | killed | none (killed on reward_std ~0) | inconclusive | abandon_line |
| exp-10 | 726 | 5.48 | grpo | exp-07 | gsm8k train questions (temperature 1.2) | 2e-6 / 400 steps | completed | 0.784 @ n=500 (metrics_grpo_exp_500.json) | inconclusive | adopt |
| exp-11 | 783 | 6.13 | grpo | exp-10 | gsm8k train + orca-math 12000 (~19.4K) | 2e-6 / 400 steps | completed | 0.808 @ n=500 (metrics_grpo_orca.json) | supported | adopt |
| exp-12 | 841 | 7.26 | grpo | exp-11 | gsm8k train + orca-math 15000 @ offset 12000 | 2e-6 / 400 steps | completed | 0.828 @ n=500 (metrics_grpo_orca2.json); 0.8097 @ n=1319 (metrics_final_full.json) | supported | adopt |
| exp-13 | 865 | 7.90 | grpo | exp-12 | gsm8k train + orca-math 15000 @ offset 27000 | 2e-6 / 400 steps | completed | 0.812 @ n=500 (metrics_grpo_orca3.json) | contradicted | reject |
| exp-14 | 927 | 9.23 | decode-config | exp-12 | none (generation_config edit) | n/a | completed | 0.8211 @ n=1319 (metrics_final_greedy_full.json) | inconclusive | adopt |

Chain that reached the packaged model: base_model -> exp-06 (few-shot SFT, `<|endoftext|>` stop token)
-> exp-07 -> exp-10 -> exp-11 -> exp-12 (weights) -> exp-14 (greedy decoding). exp-14 is the last
state of `final_model` in the stream.
