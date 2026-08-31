# Reconstructed experiment cards — gsm8k / Qwen3-1.7B-Base / 10 h / 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 23924 | 0.86 | sft | base_model | data/sft_round1 (v1: gold x4 + OM2 80k, 55/25/10/10) | 1e-5 / 1 | failed | none (OOM at step 0 of 3049) | inconclusive | iterate |
| exp-02 | 26284 | 0.88 | sft | base_model | data/sft_round1 (v1, ~220M tokens) | 1e-5 / 1 | killed | none (killed at ~12 min, 4.9 h projected) | inconclusive | abandon_line |
| exp-03 | 31594 | 1.12 | sft | base_model | data/sft_round1 (v2: gold x3 + OM2 62k, 84,335 rows / 130.4M tok) | 1e-5 / 1 | completed | accuracy 0.766 @500 (mid ckpt 0.744) | inconclusive | adopt |
| exp-04 | 41076 | 3.77 | sft | exp-03 | data/star_r1.jsonl (28,256 self-verified) + data/sft_round2 (~58k rows) | 5e-6 / 1 | completed | accuracy 0.768 @500 (+0.002 vs exp-03); 0.7536 full | inconclusive | adopt |
| exp-05 | 42663 | 4.63 | sft | base_model | data/sft_round3_150m (97k rows / 150.3M tok: gold x3 + STaR + OM2 62k) | 1e-5 / 1 | completed | accuracy 0.760 @500 (-0.006 vs exp-03) | contradicted | adopt |
| exp-06 | 46442 | 7.85 | merge | exp-03 + exp-04 | none | n/a | failed | none (save_pretrained rejected the greedy generation_config) | inconclusive | iterate |
| exp-07 | 46915 | 7.86 | merge | exp-03 + exp-04 | none | n/a | completed | accuracy 0.780 @500 (+0.012 vs exp-04) | supported | adopt |
| exp-08 | 47871 | 7.91 | merge | exp-03 + exp-04 + exp-05 | none | n/a | completed | accuracy 0.766 @500 (-0.014 vs exp-07) | contradicted | reject |
| exp-09 | 48225 | 7.95 | other (packaging) | exp-07 | none | n/a | completed | accuracy 0.7627 (n=1319, full official path); 0.727 @150 defaults | supported | adopt |
| exp-10 | 49858 | 8.05 | merge | exp-03 + exp-04 | none | n/a | completed | accuracy 0.782 @500 but 0.7589 full (-0.0038 vs exp-09) | contradicted | reject |

Submitted artifact: exp-09 — `final_model/`, a byte copy of the exp-07 uniform weight average of the
round-1 and round-2 SFT endpoints, scored 0.7627 (stderr 0.0117) on all 1319 test items.
