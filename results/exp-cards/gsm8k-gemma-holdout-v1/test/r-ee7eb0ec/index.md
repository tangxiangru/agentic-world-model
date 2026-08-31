# r-ee7eb0ec — extracted experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h, 1x H100
9 cards. The run ends on exp-08's merge in `final_model` (accuracy 0.067 @ n=30);
the best-scoring candidate was exp-06 (0.267 @ n=30), which exp-07 overwrote and
nobody restored. No per-event timestamps in the digest, so `elapsed_h` is null on
every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 63 | null | sft | base_model | gsm8k train + MetaMathQA (267419 -> 267274) | 2e-4 / 1 | failed | — | inconclusive | iterate |
| exp-02 | 75 | null | sft | base_model | gsm8k train + MetaMathQA (267274) | 2e-4 / 1 | killed | — (5738 steps at ~3.85 s/it, ~6 h projected) | inconclusive | abandon_line |
| exp-03 | 109 | null | sft | base_model | gsm8k train x3 + 40K MetaMathQA GSM (62419) | 2e-4 / 2 | completed | accuracy 0.16 @ n=50 (printed at [174], no eval json) | inconclusive | reject |
| exp-04 | 228 | null | sft | base_model | gsm8k train x3 + 15K MetaMathQA GSM (37419) | 2e-4 / 2 (max_steps 3500) | failed | — (stopped at step 1289, no checkpoint) | inconclusive | iterate |
| exp-05 | 277 | null | sft | base_model | gsm8k train x3 + 15K MetaMathQA GSM (37419) | 2e-4 / max_steps 5000 | failed | — (stopped at step 2913; checkpoint-2000, loss 0.268) | inconclusive | adopt |
| exp-06 | 345 | null | merge | exp-05 | — | — | completed | accuracy 0.267 @ n=30 (eval_v4_quick.json) | inconclusive | reject |
| exp-07 | 392 | null | sft | base_model | gsm8k train x3 + 15K MetaMathQA GSM (37419) | 5e-4 / max_steps 2300 | completed | accuracy 0.033 @ n=30 (eval_v6_quick.json, -0.233 vs exp-06) | contradicted | reject |
| exp-08 | 430 | null | sft | base_model | gsm8k train x3 + 15K MetaMathQA GSM (37419) | 3e-4 / max_steps 2300 | completed | accuracy 0.067 @ n=30 (eval_v7_quick.json, -0.200 vs exp-06) | contradicted | adopt |
| exp-09 | 480 | null | sft | base_model | same mixture as prompt/completion pairs (37419) | 3e-4 / max_steps 2300 | killed | — (66 of 2300 steps, budget ran out) | inconclusive | abandon_line |

Two cards carry `adopt`: exp-05, because its checkpoint-2000 is the parent of the
exp-06 merge, and exp-08, because its weights are the ones left in `final_model`
when the run ends. The stream never states what was submitted.

The through-line: from exp-03 on, every candidate answers correctly far more
often than it scores, because it emits "ANSWER: X" and then starts a new question
while the scorer matches the last answer in the text. The agent's own first-vs-last
answer probe [433][434][477] reads 10/30 vs 8/30 (exp-06), 1/30 vs 1/30 (exp-07)
and 12/30 vs 2/30 (exp-08). Raising the learning rate to 5e-4 bought the stop
token at the cost of the reasoning (exp-07); the completion-only-loss fix aimed at
it directly (exp-09) got 66 steps before the budget ended.

Not cards: prepare_data*.py runs; the hand-copy of preprocessor_config.json /
processor_config.json into final_model at [171], which repaired vLLM loading for
the exp-03 candidate without changing any weights; train_v5_resume.py, written at
[383] and never launched; prepare_data_v2.py / v4 outputs, built and never read by
a launch. No smoke tests or dry runs appear — exp-01, exp-02, exp-04 and exp-05
were all full-scale launches that crashed or were killed.
