# r-2cd75f43 — extracted experiment cards

All measurements are the agent's own evals of the supplied `evaluate.py` at `--limit 150`,
except the base-model comparator (0.34), which it measured once at `--limit 50`.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 92 | 0.15 | sft | base_model | gsm8k train (7473) | 1e-5 / 3 | completed | acc 0.04 @150 (agent's later note, no result event) | inconclusive | abandon_line |
| exp-02 | 151 | 0.71 | sft | base_model | gsm8k train (7473) | 1e-5 / 3 | completed | acc 0.480 @150 (0.040 before the eos fix) | inconclusive | adopt |
| exp-03 | 213 | 1.41 | other (package to final_model) | exp-02 | — | — | completed | none (copy not evaluated) | inconclusive | reject |
| exp-04 | 229 | 1.44 | sft | base_model | gsm8k train + 40k MetaMathQA GSM-derived (47473) | 1e-5 / 2.0 | completed | acc 0.480 @150 (delta 0.000 vs exp-02) | contradicted | reject |
| exp-05 | 278 | 3.06 | sft | base_model | gsm8k train (7473) | 2e-5 / 6.0 | completed | acc 0.713 @150 (delta +0.233 vs exp-02) | supported | adopt |
| exp-06 | 298 | 3.73 | other (package to final_model) | exp-05 | — | — | completed | acc 0.700 @150 (delta -0.013 vs exp-05) | inconclusive | adopt |
| exp-07 | 308 | 3.74 | sft | base_model | gsm8k train + 20k MetaMathQA GSM_Rephrased (27473) | 2e-5 / 4.0 | completed | acc 0.627 @150 (delta -0.086 vs exp-05) | contradicted | reject |
| exp-08 | 332 | 5.19 | sft | base_model | gsm8k train (7473) | 2e-5 / 8.0 | completed | acc 0.607 @150 (delta -0.106 vs exp-05) | contradicted | reject |
| exp-09 | 348 | 6.05 | sft | base_model | gsm8k train (7473) | 2e-5 / 5.0 | completed | acc 0.613 @150 (delta -0.100 vs exp-05) | contradicted | reject |
| exp-10 | 375 | 7.53 | sft | base_model | gsm8k train (7473) | 2e-5 / 7.0 | completed | acc 0.680 @150 (delta -0.033 vs exp-05) | contradicted | reject |
| exp-11 | 387 | 7.53 | sft | base_model | gsm8k train (7473) | 1.5e-5 / 6.0 | completed | acc 0.547 @150 (agent's results table, no result event) | contradicted | reject |
