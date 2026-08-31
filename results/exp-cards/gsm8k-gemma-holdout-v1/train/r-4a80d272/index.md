# Reconstructed experiment cards - run r-4a80d272

Base model: Qwen/Qwen3-4B-Base. Benchmark: gsm8k. Budget: 10 h, one H100.
21 cards, one per launch that can be pointed at an event in the digest.
Two `--skip-train` dry runs ([71], [158]) are recorded as `provenance.smoke_runs`
on exp-01 and exp-04 rather than as cards.

Measurements marked *fast* used `--max-connections 4 --max-tokens 1024
--gpu-memory-utilization 0.55`; the rest used the evaluator's defaults
(`--max-connections 2 --max-tokens 4000 --gpu-memory-utilization 0.3`).
The two settings are not interchangeable: candidate G scored 0.813 fast and
0.787 default on the same 150 samples.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 74 | 0.17 | sft | base_model | gsm8k x3 + gsm8k-exact x2 + synth 35k + orca 60k | 1.5e-4 / 1 | killed | - | inconclusive | abandon_line |
| exp-02 | 86 | 0.25 | sft | base_model | gsm8k x3 + gsm8k-exact x1 + synth 25k + orca 30k | 1.7e-4 / 1 | failed | - (CUDA OOM at step 0) | inconclusive | abandon_line |
| exp-03 | 91 | 0.29 | sft | base_model | gsm8k x3 + gsm8k-exact x1 + synth 25k + orca 30k | 1.7e-4 / 1 | completed | 0.040 @50 (eval_candidate_b2_50.json) *fast* | inconclusive | reject |
| exp-04 | 161 | 1.78 | sft | base_model | gsm8k x6 + synth 20k + orca 10k, terminator=eos | 1.8e-4 / 1 | failed | - (CUDA OOM at step 0) | inconclusive | abandon_line |
| exp-05 | 164 | 1.81 | sft | base_model | gsm8k x6 + synth 20k + orca 10k, terminator=eos | 1.8e-4 / 1 | completed | 0.727 @150 (eval_candidate_c2_150.json) *fast* | supported | reject |
| exp-06 | 299 | 2.83 | decode-config | exp-03 | - | - | completed | 0.060 @50 (eval_candidate_b2_imend_50.json) *fast* | contradicted | reject |
| exp-07 | 327 | 2.91 | sft | base_model | gsm8k x6 + gsm8k-exact x1 + synth 20k + orca 10k | 1.6e-4 / 1 | completed | 0.767 @150 (eval_candidate_e_150.json) *fast* | supported | adopt |
| exp-08 | 430 | 4.27 | other (packaging) | exp-07 | - | - | completed | 0.760 @150 (eval_final_150.json) *fast* | inconclusive | adopt |
| exp-09 | 484 | 4.36 | decode-config | exp-08 | - | - | completed | 0.793 @1319 (eval_final_default_full.json) | supported | adopt |
| exp-10 | 511 | 4.44 | sft | base_model | gsm8k x10 + gsm8k-exact x2 + synth 30k, no orca | 1.35e-4 / 1 | completed | 0.700 @50 (eval_candidate_f_50.json) *fast* | contradicted | reject |
| exp-11 | 641 | 6.44 | merge | exp-10 | adapter checkpoint-3000 | - | completed | 0.720 @50 (eval_candidate_f3000_50.json) *fast* | inconclusive | reject |
| exp-12 | 680 | 6.57 | sft | base_model | gsm8k x6 + gsm8k-exact x2 + synth 20k + orca 15k, seed 123 | 1.25e-4 / 1 | completed | 0.787 @150 (eval_candidate_g_default_150.json) | contradicted | adopt |
| exp-13 | 757 | 8.34 | merge | exp-12 | adapter checkpoint-2250 | - | completed | 0.7945 @1319 (eval_candidate_g2250_default_full.json) | supported | adopt |
| exp-14 | 828 | 8.77 | other (packaging) | exp-13 | - | - | completed | - (never evaluated in this state) | inconclusive | adopt |
| exp-15 | 831 | 8.78 | merge | exp-07 | adapter checkpoint-2500 | - | completed | 0.773 @150 (eval_candidate_e2500_default_150.json) | contradicted | reject |
| exp-16 | 836 | 8.82 | decode-config | exp-14 | - | - | completed | 0.7953 @1319 (eval_final_eos_imend_only_default_full.json) | supported | adopt |
| exp-17 | 860 | 9.04 | decode-config | exp-14 | - | - | completed | 0.7908 @1319 (eval_final_promoted_default_full.json) | contradicted | reject |
| exp-18 | 866 | 9.05 | decode-config | exp-12 | - | - | completed | 0.7801 @1319 (eval_candidate_g_imend_only_default_full.json) | contradicted | reject |
| exp-19 | 897 | 9.47 | decode-config | exp-17 | - | - | completed | 0.7915 @1319 (eval_final_restored_list_default_full.json) | inconclusive | reject |
| exp-20 | 913 | 9.66 | other (packaging) | exp-13 | - | - | completed | 0.7908 @1319 (eval_final_reset_g2250_default_full.json) | inconclusive | adopt |
| exp-21 | 939 | 9.84 | decode-config | exp-20 | - | - | completed | 0.8067 @150 (eval_final_imend_last_default_150.json) | supported | adopt |

**Submitted artifact: exp-21** - `final_model` = the exp-13 merge of the exp-12
LoRA adapter at step 2250, with the deterministic single-`<|im_end|>` EOS
generation config from exp-16.

Run-level notes:
- Two directories the run evaluated, `runs/candidate_b2_imend_eos` (exp-06) and
  `runs/candidate_g_imend_only` (exp-18), are never created by any command in
  the digest; their evaluation commands are the only events that point at them
  and stand as those cards' `launch_i`.
- The last third of the run is decoding-config churn on the submission path.
  Four full-split runs of identical weights returned 1043, 1044, 1048 and 1049
  correct out of 1319, and every promotion decision from exp-13 onward rests on
  differences of one to six questions, inside the reported +-0.011 stderr.
- The submitted artifact was never given a full-split run in its exact final
  state.
