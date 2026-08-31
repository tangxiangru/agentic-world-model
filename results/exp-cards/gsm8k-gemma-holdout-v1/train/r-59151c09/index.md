# r-59151c09 - reconstructed experiment cards

Base model: Qwen/Qwen3-4B-Base. Benchmark: gsm8k, 10 h, 1x H100.
The digest carries no timestamps, so every `elapsed_h` is null.
All measurements are the agent's own `evaluate.py` runs (inspect_evals/gsm8k,
`--limit N` of the official set); `outcome.official_accuracy` is not written.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 63 | null | sft | base_model | gsm8k-train 7,473 + MetaMathQA GSM_Rephrased/GSM_AnsAug 160,000 (167,473) | 2e-5 / 3 (killed at step 4000 of 31,404) | killed | 0.080 (n=50) | inconclusive | adopt |
| exp-02 | 97 | null | decode-config | exp-01 | - | - | completed | 0.740 (n=50, eval_v1_ckpt4000.json) | supported | adopt |
| exp-03 | 107 | null | sft | exp-01 | same 167,473 mixture (resume from checkpoint-4000) | 2e-5 / 1.0 (10,468 steps) | completed | 0.760 (n=50) / 0.733 (n=150, eval_v1_150.json) | inconclusive | adopt |
| exp-04 | 132 | null | sft | exp-03 | gsm8k-train 7,473 | 5e-6 / 5 (2,340 steps) | completed | 0.667 (n=150, eval_v2_150.json) | contradicted | reject |
| exp-05 | 161 | null | grpo | exp-03 | gsm8k-train prompts 7,473 | 5e-7 / 1.0 | killed | none | inconclusive | abandon_line |
| exp-06 | 183 | null | grpo | exp-03 | gsm8k-train prompts 7,473 | 5e-7 / max_steps 200 | killed | none | inconclusive | abandon_line |
| exp-07 | 195 | null | grpo | exp-03 | gsm8k-train prompts 7,473 | 5e-7 / max_steps 200 | killed | none | inconclusive | abandon_line |
| exp-08 | 219 | null | other (packaging to final_model) | exp-03 | - | - | completed | 0.700 (n=150, eval_final_150.json) | inconclusive | adopt |
| exp-09 | 230 | null | sft | base_model | gsm8k-train 7,473 | 2e-5 / 5 (2,340 steps, 0.65 h) | completed | 0.660 (n=150, eval_v3_150.json); 0.767 at [301] (eval_v3_greedy.json) | contradicted | reject |
| exp-10 | 291 | null | decode-config | exp-08 | - | - | completed | 0.840 (n=150, eval_final_greedy.json); 0.827 on repeat | supported | adopt |

Submitted artifact: `/home/ben/task/final_model` - the exp-03 weights packaged at
exp-08 and switched to greedy decoding at exp-10.
