# Reconstructed experiment cards

Base model: `HuggingFaceTB/SmolLM3-3B-Base` · benchmark: gsm8k · budget: 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 33772 | 0.71 | sft | base_model | packs.jsonl (122,883 pairs: OMI-2 70.5K + MetaMathQA 45K + GSM8K-train 7.4K) | 1.5e-5 / 2 | completed | 0.767 @150 (eval_final.log); 0.7718 @1319 offline | inconclusive | adopt |
| exp-02 | 40548 | 3.74 | rft | exp-01 | packs_stage2.jsonl (RFT r1 27,934 self-samples + 20K stage-1 replay) | 7e-6 / 1 | failed | none (save crashed on generation-config validation, weights lost) | inconclusive | iterate |
| exp-03 | 42608 | 4.29 | rft | exp-01 | packs_stage2.jsonl (RFT r1 27,934 self-samples + 20K stage-1 replay) | 7e-6 / 1 | completed | 0.827 @150 (+6.0 vs exp-01); 0.7945 @1319 offline | supported | adopt |
| exp-04 | 44396 | 5.04 | rft | exp-03 | packs_stage3.jsonl (RFT r2 28,350 + 10K r1 + 15K stage-1 replay) | 5e-6 / 1 | completed | 0.8021 @1319 offline (+0.0076 vs exp-03); 0.820 @150 | inconclusive | adopt |
| exp-05 | 61939 | 6.56 | other (packaging / decode-config) | exp-04 | — | — / — | completed | 0.7923 and 0.7961 @1319 official mc2; 0.820 @150 | inconclusive | adopt |
| exp-06 | 65711 | 7.30 | sft | exp-04 | packs_stage4.jsonl (fresh OMI-2 uniques + 10K RFT r2 + 8K stage-1 replay) | 3e-6 / 1 | completed | 0.8036 @1319 offline (+0.0015); 0.800 @150 (-0.02) | inconclusive | reject |

Adopted / submitted: **exp-05** — `final_model/`, byte-identical to the exp-04 checkpoint.

Not cards (no launch event in the stream): the 0.5/0.5 weight soup of exp-03 and exp-04
(evaluated at [63916] / [63964], offline 0.7991, rejected) — the command that built it does
not appear. Eleven pipeline smoke/dry runs and the baseline eval of the untrained base model
(0.19 at limit 100) are recorded on exp-01 as `provenance.smoke_runs` and as its comparator.
