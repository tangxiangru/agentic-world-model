# r-94f796fd — extracted experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100 80GB.
7 cards, one per launch. No per-event timestamps in the digest, so `elapsed_h` is null everywhere.
The 32-sample dry run at [117] is recorded as `provenance.smoke_runs` on exp-01, not as a card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 121 | null | sft (LoRA, plain format) | base_model | gsm8k train, 2048 ex | 2e-4 / 1.5 | completed | acc 0.10 @ n=50 (exp_plain_2k_50.json), -0.14 vs base 0.24 | contradicted | reject |
| exp-02 | 190 | null | sft (LoRA, plain + EOS) | base_model | gsm8k train, 2048 ex | 2e-4 / 1.0 | completed | none (no evaluate.py run; stopping probe only, [210]) | inconclusive | adopt |
| exp-03 | 213 | null | sft (LoRA, eval-fixed 10-shot) | exp-02 | gsm8k train, 2048 ex | 8e-5 / 0.5 | completed | acc 0.52 @ n=50 (exp_evalfixed10_eos_2k_50.json), +0.28 vs base 0.24 | supported | reject |
| exp-04 | 228 | null | sft (LoRA, plain, full data) | exp-02 | gsm8k train, all rows (−128 val) | 1e-4 / 1.5 | completed | acc 0.41 @ n=100 (stageA_full_plain_100.json), no comparator | inconclusive | adopt |
| exp-05 | 290 | null | sft (LoRA, eval-fixed 10-shot, full data) | exp-04 | gsm8k train, all rows (−128 val) | 5e-5 / 0.5 | completed | acc 0.64 @ n=100 (stageB_full_evalfixed10_100.json), +0.23 vs exp-04 | supported | adopt |
| exp-06 | 333 | null | sft (LoRA, low-LR anneal) | exp-05 | gsm8k train, all rows (−128 val) | 2e-5 / 0.25 | completed | none (dropped on eval_loss 0.4953 > 0.4358, [349]-[350]) | inconclusive | reject |
| exp-07 | 355 | null | other (packaging, cp -a) | exp-05 | — | — / — | completed | acc 0.5667 @ n=150 (final_model_150.json), no comparator | inconclusive | adopt |

Submitted artifact: **exp-07** — `final_model`, a byte copy of exp-05's merged checkpoint (`stageB_full_evalfixed10_merged`).
