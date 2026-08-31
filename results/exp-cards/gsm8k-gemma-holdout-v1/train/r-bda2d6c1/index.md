# r-bda2d6c1 - reconstructed experiment cards

Qwen/Qwen3-1.7B-Base on gsm8k, 10 h on one H100. 15 launches, all SFT from the base
model; exp-15 is the checkpoint the run ended on (final_model).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 65 | 0.1 | sft | base_model | gsm8k | 1e-05 / 3 | completed | 0.660 @150 | inconclusive | adopt |
| exp-02 | 194 | 0.63 | sft | base_model | gsm8k + metamath-gsm 30k | 1e-05 / 2 | completed | - | inconclusive | reject |
| exp-03 | 220 | 1.39 | sft | base_model | gsm8k | 5e-06 / 5 | completed | 0.053 @150 | contradicted | reject |
| exp-04 | 252 | 2.01 | sft | base_model | gsm8k + rejection_run1 | 1e-05 / 2 | completed | 0.693 @150 | supported | adopt |
| exp-05 | 269 | 2.57 | sft | base_model | gsm8k + rejection_run4 | 1e-05 / 2 | completed | 0.387 @150 | contradicted | reject |
| exp-06 | 283 | 3.26 | sft | base_model | gsm8k + rejection_filtered | 1e-05 / 2 | completed | 0.587 @150 | contradicted | reject |
| exp-07 | 293 | 3.57 | sft | base_model | gsm8k + rejection_combined | 1e-05 / 2 | completed | 0.707 @150 | supported | adopt |
| exp-08 | 306 | 4.12 | sft | base_model | gsm8k + rejection_v2 | 1e-05 / 2 | completed | 0.613 @150 | contradicted | reject |
| exp-09 | 373 | 4.77 | sft | base_model | gsm8k + run9_data (numina?) | 1e-05 / 1.5 | completed | 0.540 @150 | contradicted | reject |
| exp-10 | 415 | 5.73 | sft | base_model | gsm8k + rejection_combined | 1e-05 / 2.5 | completed | - | inconclusive | reject |
| exp-11 | 450 | 6.31 | sft | base_model | gsm8k + rejection_combined | 1.5e-05 / 2 | completed | 0.712 @500 | supported | adopt (re-adopted [542]) |
| exp-12 | 480 | 6.82 | sft | base_model | gsm8k + rejection_combined | 2e-05 / 2 | completed | - | inconclusive | reject |
| exp-13 | 499 | 7.32 | sft | base_model | gsm8k + rejection_combined | 1.5e-05 / 2 | completed | 0.733 @150 | inconclusive | adopt (reverted [542]) |
| exp-14 | 550 | 8.03 | sft | base_model | gsm8k + rejection_combined | 1.3e-05 / 2.2 | completed | - | inconclusive | reject |
| exp-15 | 596 | 8.64 | sft | base_model | gsm8k + rejection_run7 (23456) | 1.5e-05 / 2 | completed | 0.716 @1000 | supported | adopt |

Notes on reading this table:

- `best measurement` is the agent's own eval at the largest sample count it ran on that
  checkpoint (`accuracy @n`, inspect_evals/gsm8k through the supplied evaluate.py). `-`
  means no accuracy for that launch appears anywhere in the stream.
- Comparisons switch protocol partway: exp-01..exp-13 are compared at --limit 150,
  exp-14 and exp-15 at --limit 500. The 150-sample stderr is about +/-4 pts and the
  500-sample stderr about +/-2, so every positive delta on this table sits within
  roughly one stderr of its own eval.
- `adopt` marks the launches whose output was copied to final_model: exp-01 [183],
  exp-04 [264], exp-07 [302], exp-11 [475] and again [542], exp-13 [521], exp-15 [617].
  The copies themselves are not carded: each is a plain `cp -r` of an already-carded
  training output, verified byte-identical by md5 at [586] and [634].
- No smoke tests or dry runs appear in this run; every launch was meant as a candidate.
- The five rejection/mixture files (rejection_run1, rejection_filtered,
  rejection_combined, rejection_v2, run9_data) were built by commands that are not in
  the digest, so exp-04, exp-06..exp-14 cannot be reproduced end to end.
