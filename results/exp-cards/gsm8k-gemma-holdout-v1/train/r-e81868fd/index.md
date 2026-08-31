# r-e81868fd — reconstructed experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h · 1x H100 80GB.
The digest carries no timestamps, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 134 | null | sft | base_model | train_data_train.jsonl 266,973 (gsm8k + MetaMathQA-GSM 240k + non-GSM MetaMathQA 20k) | 2e-5 / 2.0 | killed | none | inconclusive | abandon_line |
| exp-02 | 163 | null | sft | base_model | train_data_train.jsonl 266,973 (gsm8k + MetaMathQA-GSM 240k + non-GSM MetaMathQA 20k) | 2e-5 / 1.0 | completed | accuracy 0.540 @ n=150 (eval_run1_150.json); 0.433 @ n=30 vs base 0.167 | supported | adopt |
| exp-03 | 205 | null | other (packaging) | exp-02 | none | null / null | completed | accuracy 0.500 @ n=10 (eval_final_check.json) | inconclusive | adopt |
| exp-04 | 262 | null | sft | base_model | train_data_v2_train.jsonl 464,470 (gsm8k x3 + MetaMathQA-GSM 240k + Orca-Math 192,559 + non-GSM MetaMathQA 10k) | 2e-5 / 1.0 | killed | none | inconclusive | abandon_line |
| exp-05 | 285 | null | sft | exp-03 | train_data_v3_train.jsonl 227,667 (gsm8k x5 + Orca-Math 190,802) | 5e-6 / 1.0 | completed | accuracy 0.480 @ n=150 (eval_v3_150.json), -0.060 vs exp-02 | contradicted | reject |
| exp-06 | 342 | null | sft | exp-03 | train_data_small_train.jsonl 154,230 (gsm8k x10 + MetaMathQA GSM_AnsAug 80k) | 1e-6 / 1.0 | completed | accuracy 0.513 @ n=150 (eval_v4_150.json), -0.027 vs exp-02 | contradicted | reject |

Submitted artifact: `final_model` (exp-03), the packaged copy of exp-02's checkpoint.
Not carded: three truncated pipeline runs before exp-01 ([69], [81], [89]), two mangled/duplicate
launches of exp-01's command ([98], [115]), and the crashed first launch of exp-04's command ([250]) —
all recorded as `provenance.smoke_runs`. `prepare_gsm8k_only.py` built a 314,730-example set at [334]
that was never trained on (see exp-05 `next_step`).
