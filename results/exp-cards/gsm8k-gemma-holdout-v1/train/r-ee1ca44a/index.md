# r-ee1ca44a — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 62 | 0.26 | sft | base_model | meta-math/MetaMathQA 50k | 2e-5 / 1 | completed | 0.100 @ n=10 (eval_10.json) | contradicted | adopt |
| exp-02 | 110 | 1.20 | decode-config | exp-01 | — | — / — | completed | 0.553 @ n=150 (eval_150_fixed.json) | supported | reject |
| exp-03 | 124 | 1.28 | sft | base_model | AI-MO/NuminaMath-CoT 100k | 2e-5 / 1 | completed | 0.513 @ n=150 (eval_numina_150.json) | contradicted | reject |
| exp-04 | 152 | 3.51 | sft | base_model | meta-math/MetaMathQA 395k | 2e-5 / 1 | failed | — (OOM at step 184/6172) | inconclusive | iterate |
| exp-05 | 166 | 3.78 | sft | base_model | meta-math/MetaMathQA 395k | 2e-5 / 1 | completed | 0.633 @ n=150 (eval_full_150.json) | supported | adopt |
| exp-06 | 184 | 8.90 | other | exp-05 | — | — / — | completed | 0.600 @ n=10 (eval_final_check.json) | inconclusive | adopt |

Notes: exp-06 packages exp-05's checkpoint as `final_model` and is the submitted
card. Five pre-launch trl API crashes and one shell-timeout kill are recorded as
`provenance.smoke_runs` on exp-01. The `fix_tokenizer.py` eos_token patch is a
card only at exp-02, where it was measured against an already-evaluated
checkpoint (0.100 -> 0.700 at n=10); its mechanical re-applications at [138] and
[178] are recorded on exp-03 and exp-05.
