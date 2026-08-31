# Reconstructed experiment cards — run_ref r-cfdd8de7

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · 10 h on one H100.
11 launches; the submission is exp-11 (final_model = the exp-10 checkpoint).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 143 | 0.59 | sft | base_model | train.jsonl (247,473; gsm8k + MetaMath-GSM; 100k used) | 2e-5 / 1.0 | completed | 0.120 @150 (eval_v1.json); 0.707 @150 after exp-02+exp-07 | inconclusive | adopt |
| exp-02 | 202 | 1.83 | decode-config | exp-01 | — | — / — | completed | 0.533 @150 (eval_v1_fixed.json), +0.413 vs exp-01 | supported | adopt |
| exp-03 | 210 | 1.86 | other (package) | exp-02 | — | — / — | completed | — | inconclusive | reject |
| exp-04 | 222 | 1.89 | sft | base_model | train_v2.jsonl (284,944; gsm8k x2 + MetaMath-GSM + Orca-Math 30k; 150k used) | 1e-5 / 1.0 | completed | 0.553 @150 (eval_v2.json), +0.020 vs exp-02; 0.732 @500 after exp-07 | inconclusive | adopt |
| exp-05 | 258 | 3.61 | other (package) | exp-04 | — | — / — | completed | — | inconclusive | abandon_line |
| exp-06 | 286 | 3.76 | rft | exp-04 | train_v3.jsonl (80,220; RS-from-exp-04 37,801 + gsm8k gold x3 + MetaMath-GSM 20k) | 5e-6 / 1.0 | completed | 0.553 @150 (eval_v3.json), +0.000 vs exp-04 | inconclusive | adopt |
| exp-07 | 326 | 4.60 | decode-config | exp-06 | — | — / — | completed | 0.767 @150 (eval_v3_greedy.json), +0.214 vs exp-06 | supported | adopt |
| exp-08 | 345 | 4.77 | other (package) | exp-07 | — | — / — | completed | 0.76 @50 (inspect log 11-44-11) | inconclusive | reject |
| exp-09 | 364 | 5.07 | rft | exp-07 | train_v4.jsonl (100,028; RS-from-exp-07 77,609 + gsm8k gold x3) | 3e-6 / 1.0 | killed | — | inconclusive | abandon_line |
| exp-10 | 390 | 6.12 | rft | exp-07 | train_v4.jsonl (100,028; RS-from-exp-07 77,609 + gsm8k gold x3) | 3e-6 / 1.0 | completed | 0.748 @500 (eval_v4_500.json), +0.010 vs exp-07 | inconclusive | adopt |
| exp-11 | 423 | 7.13 | other (package) | exp-10 | — | — / — | completed | 0.743 @1000 (eval_final_1000.json) | inconclusive | adopt |

Smoke runs, folded onto exp-01 rather than carded: [111], [114], [119], [122] (batch-size probes on 200-400 examples) and [128] (5,000-example pipeline check, evaluated at 0.260 @50 [135]).
