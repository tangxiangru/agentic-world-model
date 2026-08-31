# r-628a7a20 — extracted experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 20368 | 0.87 | sft | base_model | data/sft_v1.jsonl (137,396) | 2e-5 / 2 | killed | none | inconclusive | abandon_line |
| exp-02 | 25061 | 1.29 | sft | base_model | data/sft_v1b.jsonl (122,397) | 2e-5 / 2 | completed | 0.673 @150 (ep2; ep1 0.620) | inconclusive | adopt |
| exp-03 | 29588 | 4.94 | rft | exp-02 | data/rft_samples.jsonl (57,010) → data/sft_v2.jsonl (81,982) | 8e-6 / 1 | failed | none (weights lost at save) | inconclusive | abandon_line |
| exp-04 | 30851 | 6.20 | rft | exp-02 | data/rft_samples.jsonl (57,010) → data/sft_v2.jsonl (81,982) | 8e-6 / 1 | completed | 0.740 @150 (+0.067 vs exp-02) | supported | adopt |
| exp-05 | 31745 | 7.50 | other (packaging) | exp-04 | — | — / — | completed | 0.7036 @1319 (full test) | inconclusive | adopt |
| exp-06 | 33171 | 7.93 | rft | exp-05 | data/rft2_samples.jsonl (57,434) → data/sft_v3.jsonl (54,080) | 5e-6 / 1 | completed | 0.727 @150 (−0.013 vs exp-04) | inconclusive | reject |

Smoke runs (not cards, recorded on exp-01): [18066] crashed with CUDA OOM at mbs 8/accum 8; [18704] passed (6K examples, 1 epoch, 0.687 @150 after export).
