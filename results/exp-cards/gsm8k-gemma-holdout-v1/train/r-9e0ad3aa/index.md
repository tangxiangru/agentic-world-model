| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 82 | 0.19 | sft | base_model | MetaMathQA GSM slice (237600) | 2e-5 / 1 | failed | none | inconclusive | iterate |
| exp-02 | 88 | 0.21 | sft | base_model | MetaMathQA GSM slice (237600) | 2e-5 / 1 | completed | accuracy 0.030 (n=100, eval_metrics.json) | inconclusive | reject |
| exp-03 | 134 | 2.38 | sft | base_model | openai/gsm8k train (7473), 5 few-shots/example | 2e-5 / 3 | completed | accuracy 0.040 (n=100, eval_metrics_v2.json) | inconclusive | adopt |
| exp-04 | 180 | 3.15 | decode-config | exp-03 | none (EOS patch on final_model_v2) | n/a | completed | accuracy 0.400 (n=100, from [181]; file overwritten) | supported | reject |
| exp-05 | 190 | 3.20 | sft | base_model | MetaMathQA GSM slice (239000) + fixed 8-shot gsm8k prefix | 1e-5 / 1 | completed | accuracy 0.020 (n=100, eval_metrics_v4.json) | contradicted | adopt |
| exp-06 | 214 | 7.30 | decode-config | exp-05 | none (EOS patch on final_model) | n/a | completed | accuracy 0.030 (n=100, eval_metrics_v5.json) | inconclusive | adopt |
