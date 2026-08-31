# r-2354e591 — HuggingFaceTB/SmolLM3-3B-Base on gsm8k, 10 h, 1x H100

12 cards. The digest carries no timestamps, so `elapsed_h` is null on every card.
Submission: **exp-10** — the exp-09 merge (v2) copied into `final_model/`, md5-verified identical at [350].

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 106 | null | sft (LoRA r=64) | base_model | gsm8k x5 + MetaMathQA 60k GSM/15k MATH | 2e-4 / 3 | failed | — | inconclusive | abandon_line |
| exp-02 | 130 | null | sft (LoRA r=64) | base_model | gsm8k x5 + MetaMathQA 60k/15k | 2e-4 / 3 | failed | — | inconclusive | abandon_line |
| exp-03 | 146 | null | sft (LoRA r=64) | base_model | gsm8k x5 + MetaMathQA 60k/15k | 2e-4 / 3 | killed | — | inconclusive | abandon_line |
| exp-04 | 160 | null | sft (LoRA r=64) | base_model | gsm8k x3 + MetaMathQA 30k/5k | 2e-4 / 2 | completed | — (scored via exp-05) | inconclusive | adopt |
| exp-05 | 195 | null | merge | exp-04 | — | — | completed | 0.587 @150 (eval_run1.json); 0.533 @30 | inconclusive | reject |
| exp-06 | 217 | null | sft (full FT) | base_model | gsm8k x5 + MetaMathQA 60k/10k (106,789 rows) | 2e-5 / 3 | killed | — | inconclusive | abandon_line |
| exp-07 | 228 | null | sft (full FT) | base_model | gsm8k x3 + MetaMathQA 20k/5k | 2e-5 / 2 | completed | 0.587 @150 (eval_run2.json), delta 0.000 vs exp-05 | contradicted | reject |
| exp-08 | 265 | null | sft (LoRA r=128, packing) | base_model | gsm8k x5 (<<>> kept) + MetaMathQA 30k/5k + Orca-Math 15k | 2e-4 / 3 | completed | — (scored via exp-09) | inconclusive | adopt |
| exp-09 | 289 | null | merge | exp-08 | — | — | completed | 0.613 @150 (eval_run3.json), +0.027 vs exp-07; 0.667 on an unsaved re-run | supported | adopt |
| exp-10 | 299 | null | other (copy to final_model) | exp-09 | — | — | completed | 0.540 @150 (eval_final.json), -0.073 vs exp-09 on identical weights | inconclusive | adopt |
| exp-11 | 304 | null | sft (LoRA r=64 stacked) | exp-09 | gsm8k x8 + MetaMathQA-GSM 20k (seed 123) | 5e-5 / 2 | completed | — (scored via exp-12) | inconclusive | adopt |
| exp-12 | 328 | null | merge (onto exp-09) | exp-11 | — | — | completed | 0.613 @150 (eval_run4.json), delta 0.000 vs exp-09; 0.587 on an unsaved re-run | contradicted | reject |

Notes carried on the cards:

- No baseline: the pretrained base model was never evaluated, so no card has a
  base-model comparator and the first candidate's 0.587 stands alone.
- All evaluation is `evaluate.py --limit 150` (inspect_evals/gsm8k through vllm),
  no seed, temperature sampling. The same weights scored 0.540, 0.613 and 0.667
  across three runs of that command, so every delta in this run is inside the
  measurement's own spread.
- No smoke tests or dry runs appear in the stream; exp-01 and exp-02 were
  full-size launches that died on the trl API, and exp-03 and exp-06 were
  full-size launches that stopped without producing a checkpoint.
