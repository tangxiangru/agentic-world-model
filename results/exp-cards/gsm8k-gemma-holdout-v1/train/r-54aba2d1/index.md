# r-54aba2d1 — reconstructed experiment cards (train side)

base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100
7 cards, one per launch. This stream carries turn timestamps, so `elapsed_h` is filled from the `t=+H.HHh` header of the launch event.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 64 | 0.14 | sft (full-parameter) | base_model | gsm8k train, 6726 ex | 2e-5 / 3 | failed | none | inconclusive | iterate |
| exp-02 | 74 | 0.15 | sft (LoRA r=64) | base_model | gsm8k train, 6726 ex | 2e-4 / 3 | failed | none | inconclusive | iterate |
| exp-03 | 80 | 0.15 | sft (LoRA r=64) | base_model | gsm8k train, 6726 ex | 2e-4 / 3 | completed | none (train_loss 0.077, 1263 steps) | inconclusive | adopt |
| exp-04 | 94 | 0.73 | merge | exp-03 | none | null / null | completed | accuracy 0.406 (n=1319), eval_all.json | inconclusive | adopt |
| exp-05 | 136 | 1.04 | sft (LoRA r=128) | base_model | gsm8k train, 6726 ex | 1e-4 / 5 | completed | none (train_loss 0.053, 2105 steps) | inconclusive | adopt |
| exp-06 | 142 | 2.00 | merge | exp-05 | none | null / null | completed | accuracy 0.287 (n=150), eval_run2_150.json | contradicted | reject |
| exp-07 | 148 | 2.06 | merge (repackage) | exp-03 | none | null / null | completed | accuracy 0.409 (n=1319), eval_final_full.json | supported | adopt |

Submitted artifact: exp-07 — `rm -rf final_model && cp -r output/run1 final_model_temp && python3 merge_lora.py final_model_temp final_model` at [148], scored 0.400 on 150 samples [152] and 0.4094 on all 1319 [155]. It is the last write to `final_model` in the stream; `final_model_run2` and `final_model_temp` are left beside it.

Run-level notes:
- The base-model baseline launched at [55] never returned a number and `baseline_eval.json` is not in the workspace, so no card has a base-model comparator.
- Only two distinct recipes were ever trained (3 epochs at LoRA r=64/2e-4, and 5 epochs at r=128/1e-4); the second overfits (eval_loss rises every epoch after the first) and loses 13 pts on the same 150-sample slice.
- `prepare_data_v2.py` (multi-dataset mixture) was written, fixed after a syntax error and run at [130], but the digest has no result block for it and no launch ever reads `training_data_v2/`.
- The run ends at [165] with 7 h 44 m of the 10 h budget unspent [163].
- Console output for both training launches is truncated in the digest; loss, step and eval_loss figures come from `training_run1.log` / `training_run2.log` in the workspace snapshot.
