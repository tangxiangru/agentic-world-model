| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 287 | 0.17 | sft | base_model | work/gsm8k_train.jsonl | 1e-5 / 3 | completed | 0.427 acc @150 (logs/sft1.json) | inconclusive | adopt |
| exp-02 | 350 | 0.52 | decode-config | exp-01 | — | — / — | completed | 0.327 acc @150 (logs/sft1_greedy.json) | contradicted | reject |
| exp-03 | 401 | 0.58 | sft | base_model | work/mix.jsonl (gsm8k x3 + ~90k MetaMathQA) | 1e-5 / 2 | completed | 0.287 acc @150 (logs/sft2_e2_greedy.json) | contradicted | reject |
| exp-04 | 720 | 4.96 | sft | base_model | work/gsm8k_fewshot.jsonl | 1e-5 / 2 | failed | none (CUDA OOM) | inconclusive | iterate |
| exp-05 | 753 | 5.00 | sft | base_model | work/gsm8k_fewshot.jsonl | 1e-5 / 2 | completed | 0.480 acc @150 (logs/sft3_e2_greedy.json) | supported | adopt |
| exp-06 | 829 | 6.90 | other (packaging) | exp-05 | — | — / — | completed | 0.487 acc @150 (logs/final_confirm.json) | inconclusive | adopt |
| exp-07 | 848 | 6.91 | sft | base_model | work/mix_fewshot.jsonl (gsm8k x2 + 18k MetaMathQA, few-shot wrapped) | 1e-5 / 1 | completed | 0.407 acc @150 (logs/sft4_greedy.json) | contradicted | reject |
