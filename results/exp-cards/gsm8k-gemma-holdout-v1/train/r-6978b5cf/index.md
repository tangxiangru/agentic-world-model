# r-6978b5cf — reconstructed experiment cards

Base model Qwen/Qwen3-4B-Base, gsm8k, 10 h budget, one H100. 18 cards.
Comparators used by the agent: base_model 0.480 @100 (baseline.json), 0.393 @150 (baseline_full.json).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 54 | 0.16 | sft | base_model | gsm8k-train 7473 (chat template) | 2e-4 / 3 ep | killed (423/1404, no ckpt) | — | inconclusive | abandon_line |
| exp-02 | 72 | 0.37 | sft | base_model | gsm8k-train 7473 (chat template) | 2e-4 / 3 ep | killed (364/702, ckpt-200) | — | inconclusive | adopt |
| exp-03 | 80 | 0.54 | merge | exp-02 | — | — | completed | 0.190 @100 (eval_v1.json), −0.290 | contradicted | reject |
| exp-04 | 96 | 0.63 | sft | base_model | gsm8k-train 7473 (chat template) | 5e-4 / 1 ep | completed | — | inconclusive | adopt |
| exp-05 | 98 | 0.78 | merge | exp-04 | — | — | completed | 0.170 @100 (eval_v2.json), −0.310 | contradicted | reject |
| exp-06 | 120 | 0.85 | sft | base_model | gsm8k-train 7473 (Reasoning/ANSWER in chat template) | 5e-4 / 2 ep | killed (537/936, ckpt-500) | — | inconclusive | adopt |
| exp-07 | 128 | 1.03 | merge | exp-06 | — | — | completed | 0.200 @100 (eval_v3.json), −0.280 | contradicted | reject |
| exp-08 | 140 | 1.17 | sft | base_model | gsm8k-train 7473 (eval prompt template) | 2e-4 / 2 ep | killed (452/936, no ckpt) | — | inconclusive | abandon_line |
| exp-09 | 158 | 1.40 | sft | base_model | gsm8k-train 7473 (eval prompt template) | 1e-4 / 1 ep | killed (142/234, no ckpt) | — | inconclusive | abandon_line |
| exp-10 | 170 | 1.58 | sft | base_model | gsm8k-train 7473 (bare Reasoning/ANSWER) | 5e-5 / 100 steps | completed | — | inconclusive | adopt |
| exp-11 | 172 | 1.64 | merge | exp-10 | — | — | completed (→ final_model) | 0.387 @150 (eval_final.json), −0.007 | inconclusive | reject |
| exp-12 | 196 | 1.81 | sft | base_model | gsm8k-train 7473, 3-shot prefixed | 3e-5 / 50 steps | completed | — | inconclusive | adopt |
| exp-13 | 198 | 1.88 | merge | exp-12 | — | — | completed | 0.140 @50 (eval_fewshot.json), no comparator | inconclusive | reject |
| exp-14 | 214 | 2.15 | sft | base_model | gsm8k-train 7473 (bare Reasoning/ANSWER) | 1e-4 / 1 ep | completed | — | inconclusive | adopt |
| exp-15 | 216 | 2.30 | merge | exp-14 | — | — | completed (→ final_model) | 0.420 @150 (eval_v2_final.json), +0.027 | inconclusive | adopt |
| exp-16 | 232 | 2.38 | sft | base_model | metamath-gsm ≤80k + gsm8k 7473 | 2e-4 / 1 ep | killed (~113/5468, no ckpt) | — | inconclusive | abandon_line |
| exp-17 | 262 | 2.81 | sft | base_model | metamath-gsm 40k + gsm8k 7473 | 2e-4 / 1 ep | killed (500/2968, no ckpt) | — | inconclusive | abandon_line |
| exp-18 | 285 | 3.59 | sft | base_model | metamath-gsm 40k + gsm8k 7473 | 2e-4 / 1 ep | killed (505/2968, no ckpt) | — | inconclusive | abandon_line |

Notes

- exp-15 is the last merge into `final_model` that the stream shows, at 0.420 on 150 items — the +2.7 points the agent reports over the base line [224] is 0.67 stderr on that protocol, so the card records it as inconclusive.
- Eight launches are folded into cards as `provenance.smoke_runs` rather than counted: the two trl API crashes before exp-01, the two OOM crashes before exp-02, the gradient-checkpointing crash before exp-09, and the three formatting crashes before exp-12. The four identical MetaMathQA relaunches at [235], [241], [246] and [250] are recorded inside exp-16.
- The digest ends at [292], t=+3.77h of a 10 h budget, with exp-18 still training.
