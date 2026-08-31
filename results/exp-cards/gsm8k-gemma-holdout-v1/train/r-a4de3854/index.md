# r-a4de3854 - extracted experiment cards

base model: Qwen/Qwen3-1.7B-Base | benchmark: gsm8k | budget: 10 h, 1x H100 | 9 cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 62 | 0.38 | sft | base_model | train.jsonl 56973 (gsm8k 6973 + MetaMathQA 50000), "Question:/Answer: ... #### N" | 2e-4 / 1 | killed (58/1781 steps, foreground timeout) | none | inconclusive | abandon_line |
| exp-02 | 66 | 0.43 | sft | base_model | train.jsonl 56973 (gsm8k 6973 + MetaMathQA 50000), "Question:/Answer: ... #### N" | 2e-4 / 1 | completed (1781 steps, 0.58 h) | 0.153 @150 (results_retry.json; 0.033 @150 in results.json before the serving-context fix) | inconclusive | reject |
| exp-03 | 158 | 3.26 | sft | base_model | train.jsonl 56973, "Question: {q}\nReasoning:\n{cot}\nANSWER: {r}" | 2e-4 / 1 | completed | 0.220 @150 (results_v2.json), +0.067 vs exp-02 | supported | reject |
| exp-04 | 194 | 3.96 | sft | base_model | train.jsonl 56973, "{q}\nReasoning:\n{cot}\nANSWER: {r}" | 2e-4 / 1 | completed | 0.133 @150 (results_v3.json), -0.087 vs exp-03 | contradicted | reject |
| exp-05 | 240 | 4.93 | sft | base_model | train.jsonl 56973, "{q}\n\nReasoning:\n{cot}\n\nANSWER: {r}" | 2e-4 / 1 | completed | 0.167 @150 (results_v4.json), +0.033 vs exp-04 | inconclusive | adopt |
| exp-06 | 278 | 5.71 | grpo | exp-05 | openai/gsm8k train 7473, prompt "{q}\n\nReasoning:", numeric-match reward | 1e-6 / 1 | killed (14/1868 steps, too slow) | none | inconclusive | abandon_line |
| exp-07 | 288 | 5.77 | grpo | exp-05 | openai/gsm8k train 7473, prompt "{q}\n\nReasoning:", numeric-match reward | 1e-6 / 1 | killed (1305/1868 steps, 2.79 h) | none | inconclusive | adopt |
| exp-08 | 322 | 8.57 | other (packaging) | exp-07 | none (cp checkpoint-1300 -> final_model) | n/a | completed | 0.120 @150 (results_grpo.json), -0.047 vs exp-05 | contradicted | reject |
| exp-09 | 368 | 8.83 | sft | base_model | train.jsonl 56973, exp-03 format restored ("Question: ... ANSWER:") | 2e-4 / 1 | completed (1781 steps, 0.59 h) | none - never evaluated | inconclusive | adopt |

Notes: all measurements are the agent's own `evaluate.py --limit 150` runs (official gsm8k items, vLLM-served
checkpoint); exp-02's two numbers are the same checkpoint before and after the serving context was raised to
16384 and generation capped at 1024 tokens. Three trainer-API crashes before exp-01 are recorded as
`provenance.smoke_runs` on exp-01, not as cards. exp-09 holds the model that final_model contained when the
stream ends at +9.55 h; it was never scored.
