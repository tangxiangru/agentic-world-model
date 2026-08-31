# r-ae9c94e6 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 79 | null | sft | base_model | openai/gsm8k (7473) + meta-math/MetaMathQA GSM (30000) | 2e-4 / 3 | completed | accuracy 0.233 @ n=30 (eval_result_quick.json) | inconclusive | adopt |
| exp-02 | 127 | null | sft | base_model | openai/gsm8k x3 (22419) + meta-math/MetaMathQA GSM (50000) | 2e-5 / 4 (killed at ~2.4) | killed | accuracy 0.000 @ n=20 (eval_result_v2_quick.json) | inconclusive | abandon_line |
| exp-03 | 173 | null | merge | exp-01 | none (merge of exp-01 epoch-3 adapter) | null / null | completed | accuracy 0.200 @ n=20 (eval_result_v1_verify.json) | inconclusive | reject |
| exp-04 | 188 | null | merge | exp-01 | none (merge of exp-01 epoch-2 adapter) | null / null | completed | accuracy 0.327 @ n=150 (eval_final.json) | inconclusive | adopt |
| exp-05 | 193 | null | merge | exp-01 | none (merge of exp-01 epoch-1 adapter) | null / null | completed | accuracy 0.200 @ n=30 (eval_v1_ep1.json) | inconclusive | reject |

Notes: the digest carries no event timestamps ("no timestamps in this run"), so
`elapsed_h` is null on every card. exp-04 is the adopted checkpoint — its output
was copied over `final_model` at [199] and nothing later in the stream changes
it. Four crashed launches ([64], [71], [115], [121]), each sharing its argv with
the real launch that followed it and none of which reached a training step, are
recorded as `provenance.smoke_runs` on exp-01 and exp-02 rather than as cards.
