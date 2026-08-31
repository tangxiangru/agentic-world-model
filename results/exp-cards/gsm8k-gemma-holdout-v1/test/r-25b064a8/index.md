# r-25b064a8 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 234 | 0.37 | sft | base_model | gsm8k.jsonl 7473 x2 + metamath_gsm.jsonl 45000 | 1e-5 / 2 | completed | 0.06 @150 greedy (same sampled) | inconclusive | adopt |
| exp-02 | 420 | 1.83 | sft | exp-01 | fewshot.jsonl (18k MetaMath, K=2-10) | 1e-5 / 2 | killed | none (killed at ~3 min for throughput) | inconclusive | abandon_line |
| exp-03 | 432 | 1.89 | sft | exp-01 | fewshot.jsonl (18k MetaMath, K=2-10) | 1e-5 / 1 | killed | none (never reappears; reissued at 443) | inconclusive | abandon_line |
| exp-04 | 443 | 1.91 | sft | exp-01 | fewshot.jsonl (18k MetaMath, K=2-10) | 1e-5 / 1 | completed | 0.70 @150, 0.684 @500, 0.7133 @150 default | supported | reject |
| exp-05 | 480 | 2.75 | other (package to final_model) | exp-04 | none | n/a | completed | none on the copy itself | inconclusive | reject |
| exp-06 | 594 | 2.97 | sft | exp-01 | fewshot_big.jsonl 44073 (40k MetaMath) | 1e-5 / 1 | completed | 0.72 @150, 0.676 @500, 0.7067 @150 default | contradicted | reject |
| exp-07 | 649 | 4.46 | sft | exp-01 | fewshot_big.jsonl 44073 + orca.jsonl 40000 | 1e-5 / 1 | completed | 0.688 @500, 0.72 @150, 0.7067 @150 default | contradicted | reject |
| exp-08 | 682 | 6.56 | other (package to final_model) | exp-07 | none | n/a | completed | 0.7067 @150 default settings | inconclusive | reject |
| exp-09 | 682 | 6.56 | sft | exp-01 | fewshot.jsonl + orca.jsonl 25000 | 1e-5 / 2 | completed | 0.7267 @150 default, 0.7133 @150, 0.686 @500 | inconclusive | adopt |
| exp-10 | 767 | 9.09 | other (package to final_model) | exp-09 | none | n/a | completed | 0.72 @150 default settings | supported | adopt |

Notes: `@150` / `@500` are the agent's own `evaluate.py --limit N` runs with
`--max-connections 8`; `default` means evaluate.py's own settings
(`--limit 150`, `max_connections 2`, `max_tokens 4000`), which the agent treated
as the graded configuration. exp-10 holds the submitted artefact
(`/home/ben/task/final_model`).
