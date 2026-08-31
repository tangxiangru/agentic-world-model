# r-aa43ab4f — reconstructed experiment cards

Base model post-trained: google/gemma-3-4b-pt · benchmark gsm8k · 10 h budget · 1x H100.
Six launches carded. One smoke test ([89]) is folded into exp-01. Base-model baseline
(0.000 on --limit 50, [44]/[68]) is not a card: it produced no candidate, it is exp-01's
comparator.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 96 | 0.15 | sft | base_model | data/gsm8k_train.jsonl (7,473; openai/gsm8k train) | 1e-5 / 3.0 | completed | 0.360 @ n=150 | inconclusive | adopt |
| exp-02 | 147 | 0.66 | sft | base_model | data/metamath_gsm.jsonl (247,473; MetaMathQA GSM_* + gsm8k train) | 1e-5 / 1.0 | completed | 0.027 @ n=150 | contradicted | adopt |
| exp-03 | 221 | 4.98 | sft | exp-02 | data/gsm8k_fewshot.jsonl (7,473; gsm8k train, 3-shot context) | 5e-6 / 2.0 | completed | 0.493 @ n=150 | supported | adopt |
| exp-04 | 245 | 5.58 | sft | exp-03 | data/fewshot_metamath_40k.jsonl (40,000; MetaMathQA GSM_* + gsm8k train, 3-shot context) | 5e-6 / 1.0 | completed | 0.660 @ n=150 | supported | adopt |
| exp-05 | 264 | 7.04 | sft | exp-04 | data/v5_data.jsonl (unknown — never built or described in the stream) | 3e-6 / 1.0 | completed | 0.580 @ n=150 | contradicted | reject |
| exp-06 | 281 | 8.11 | other (packaging) | exp-04 | — | — / — | completed | 0.656 @ n=1319 | inconclusive | adopt |

Submitted checkpoint: exp-06 — `cp -r runs/sft_v4 final_model` at [281], the last event in
the stream that touches final_model; its weights are exp-04's.

`adopt` on exp-01/exp-02/exp-03 records that each output became final_model (exp-01 at
[157], exp-03 at [231]) or the parent of a later card (exp-02 -> exp-03, exp-03 -> exp-04),
per the decision rule; only exp-06 carries the submission.
