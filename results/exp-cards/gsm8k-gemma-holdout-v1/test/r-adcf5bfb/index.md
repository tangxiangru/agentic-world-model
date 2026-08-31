# r-adcf5bfb — gsm8k / google/gemma-3-4b-pt / 10 h / 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 797 | 2.95 | rft | base_model | sft_r1.jsonl, self-sampled from base (15584; 89.8% train coverage) | 1e-5 / 2 | completed | accuracy 0.347, n=150 (eval_r1.json, official --limit 150) | supported | reject |
| exp-02 | 941 | 3.42 | rft | base_model | sft_r2.jsonl, same pool + few-shot prefixes (15584) | 1e-5 / 2 | completed | accuracy 0.520, n=250 (eval_local.py chat, printed at [1031]) | supported | iterate |
| exp-03 | 1146 | 5.60 | rft | base_model | sft_r3.jsonl, round-1 pool merged with on-policy samples from exp-02 (17759; 93.3% coverage) | 1e-5 / 2 | completed | accuracy 0.603, n=300 (eval_local.py chat, printed at [1172]); 0.600 over 900 | inconclusive | adopt |
| exp-04 | 1173 | 7.02 | other (packaging) | exp-03 | — | — / — | completed | accuracy 0.547, n=150 (eval_final_r3.json, official --limit 150) | supported | adopt |
| exp-05 | 1199 | 7.06 | decode-config | exp-04 | — | — / — | completed | accuracy 0.580, n=150 (eval_greedy.json, official --limit 150; 0.573 on re-verification, eval_FINAL.json) | supported | adopt |
| exp-06 | 1324 | 7.14 | grpo | exp-03 | openai/gsm8k train prompts, 35% with the eval's 10-shot prefix | 1e-6 / — (91 of 600 steps) | killed | accuracy 0.599, n=900 (eval_local.py chat, printed at [1423]) | inconclusive | reject |

Submitted checkpoint: `final_model` — the exp-03 weights, packaged by exp-04 and
switched to greedy decoding by exp-05, which is the last launch that wrote to
that directory. exp-01 through exp-03 all trained from the base model; only the
training data changed between them. The three official-eval numbers
(0.347, 0.547, 0.580/0.573) share the `--limit 150` protocol and are comparable;
the 0.448 / 0.520 / 0.603 / 0.600 / 0.599 figures come from the agent's own
local replica of the scorer at different limits, were printed to stdout only,
and no file in the workspace holds them.
