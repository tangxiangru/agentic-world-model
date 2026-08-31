# r-928e1ff3 — reconstructed experiment cards

base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h on one H100 · 12 launches

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 299 | 0.08 | decode-config | base_model | — | — | completed | 0.287 acc @150 | supported | adopt |
| exp-02 | 501 | 0.52 | sft | base_model | work/sft_pc.jsonl (206,785; OpenMathInstruct-2) | 1.5e-5 / 1 ep | completed | 0.822 acc @500 | supported | adopt |
| exp-03 | 648 | 1.82 | grpo | exp-02 | gsm8k train prompts | 1e-6 / 220 steps | failed | — | inconclusive | abandon_line |
| exp-04 | 654 | 1.83 | grpo | exp-02 | gsm8k train prompts | 1e-6 / 220 steps | killed | — | inconclusive | abandon_line |
| exp-05 | 732 | 1.89 | grpo | exp-02 | gsm8k train prompts | 1e-6 / 240 steps | killed | — | inconclusive | abandon_line |
| exp-06 | 822 | 2.34 | grpo | exp-02 | gsm8k train prompts | 3e-6 / 240 steps | failed | — | inconclusive | abandon_line |
| exp-07 | 894 | 2.66 | other (package) | exp-02 | — | — | completed | — | inconclusive | adopt |
| exp-08 | 896 | 2.66 | grpo | exp-02 | gsm8k train prompts | 3e-6 / 220 steps (68 run) | killed | 0.8467 acc @300 | inconclusive | adopt |
| exp-09 | 1055 | 4.18 | other (package) | exp-08 | — | — | completed | — | inconclusive | adopt |
| exp-10 | 1057 | 4.18 | grpo | exp-08 | gsm8k train prompts | 3e-6 / 170 steps (≥60 run) | failed | 0.84 acc @300 | inconclusive | adopt |
| exp-11 | 1155 | 6.33 | grpo | exp-10 | gsm8k train prompts | 2e-6 / 150 steps | completed | 0.8525 acc @400 | inconclusive | adopt |
| exp-12 | 1242 | 8.01 | other (package) | exp-11 | — | — | completed | 0.84 acc @150 · 0.8325 acc @800 | inconclusive | adopt |

Submitted artifact: **exp-12** (`/home/ben/task/final_model`, a copy of `work/rl_v4`).

Not cards — smoke and throughput runs, listed on exp-02 as `provenance.smoke_runs`: [378], [398], [413], [429].
