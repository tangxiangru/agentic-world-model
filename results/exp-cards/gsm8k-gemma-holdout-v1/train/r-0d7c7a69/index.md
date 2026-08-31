# Reconstructed experiment cards — r-0d7c7a69 (gsm8k, Qwen/Qwen3-1.7B-Base, 10 h, 1x H100)

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 368 | 0.62 | sft | base_model | sft_round1.jsonl (239,840: OpenMathInstruct-2 gsm8k-sourced 220K + gsm8k train x3) | 2e-5 / 2 ep | failed | none (OOM at bs 16) | inconclusive | abandon_line |
| exp-02 | 392 | 0.75 | sft | base_model | sft_round1.jsonl (239,840) | 2e-5 / 2 ep | completed | acc 0.7991 (n=1319, my_eval full test) | inconclusive | adopt |
| exp-03 | 533 | 3.49 | decode-config | exp-02 | none | n/a | completed | none | inconclusive | adopt |
| exp-04 | 582 | 3.84 | grpo | exp-02 | gsm8k train prompts (7,473) | 2e-6 / 160 steps | completed | acc 0.8196 (n=1319), +0.0205 vs exp-02 | supported | adopt |
| exp-05 | 636 | 4.71 | grpo | exp-04 | gsm8k train prompts (7,473) | 2.5e-6 / 300 steps | completed | acc 0.8347 (n=1319), +0.0151 vs exp-04 | supported | adopt |
| exp-06 | 700 | 6.29 | grpo | exp-05 | gsm8k train prompts (7,473) | 2.5e-6 / 240 steps | completed | acc 0.8332 at ckpt-120, final 0.8218 (n=1319), -0.0129 vs exp-05 | contradicted | reject |
| exp-07 | 753 | 7.56 | merge | exp-05 (+ exp-06 ckpt-120) | none | n/a | completed | acc 0.8264 (n=1319), -0.0083 vs exp-05 | contradicted | reject |
| exp-08 | 762 | 7.59 | decode-config | exp-05 | none | n/a | completed | acc 0.8533 (n=150, evaluate.py) | inconclusive | adopt |
| exp-09 | 772 | 7.64 | grpo | exp-05 | gsm8k train prompts (7,473) | 1e-6 / 100 steps | completed | acc 0.8355 (n=1319), +0.0008 vs exp-05 | inconclusive | adopt |
| exp-10 | 814 | 8.24 | decode-config | exp-09 | none | n/a | completed | acc 0.8533 (n=150), +0.0667 vs the same weights unpinned | supported | adopt |
| exp-11 | 826 | 8.28 | decode-config | exp-09 | none | n/a | completed | acc 0.8533 / 0.84 (n=150, two verification runs) | inconclusive | adopt |

Submitted: **exp-11** — final_model = the round-4 GRPO checkpoint exported with `generation_config.json` pinning temperature 0.0.
