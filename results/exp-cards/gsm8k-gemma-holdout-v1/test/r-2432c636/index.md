# r-2432c636 — reconstructed experiment cards

Base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on 1x H100 80GB.
12 cards, one per launch. Two smoke runs ([143], [149]) are recorded on exp-01
rather than as cards. Merge / greedy-decoding-config / eval commands are recorded
inside the card of the training launch whose output they package, not as separate
cards. Every measurement is the agent's own eval via `evaluate.py`
(inspect_evals/gsm8k, 10-shot prefix, fewshot_seed 42); `--limit` differs between
rounds and is given per card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 154 | 0.20 | sft | base_model | gsm8k-train (7473) | 2e-4 / 2.0 ep | completed | 0.06 @ n=50 | inconclusive | abandon_line |
| exp-02 | 398 | 1.34 | sft | base_model | TemplateGSM (20000) | 3e-4 / 1.0 ep | completed | none | inconclusive | adopt |
| exp-03 | 494 | 2.01 | sft | exp-02 | gsm8k-train (7473) | 1e-4 / 1.0 ep | completed | 0.22 @ n=50 | supported | adopt |
| exp-04 | 642 | 2.39 | sft | exp-03 | gsm8k-train 10-shot (7463) | 5e-5 / 1.0 ep | completed | 0.28 @ n=50 | supported | adopt |
| exp-05 | 875 | 3.50 | sft | exp-04 | TemplateGSM-7473-1k (80000) | 8e-5 / 1.0 ep | killed | none | inconclusive | abandon_line |
| exp-06 | 950 | 3.65 | sft | exp-04 | TemplateGSM-7473-1k (80000, cap 1000 steps) | 8e-5 / 1000 steps | killed | none | inconclusive | abandon_line |
| exp-07 | 973 | 3.71 | sft | exp-04 | TemplateGSM-7473-1k (80000, cap 375 steps) | 8e-5 / 375 steps | completed | 0.25 @ n=20 | inconclusive | adopt |
| exp-08 | 1227 | 4.18 | sft | exp-07 | gsm8k-train (7473) + MetaMathQA GSM_Rephrased/GSM_AnsAug (12000) | 7e-5 / 400 of 1218 steps | killed | 0.35 @ n=20; 0.30 @ n=50 | supported | adopt |
| exp-09 | 1480 | 4.47 | sft | exp-08 | gsm8k-train 10-shot (7463) | 3e-5 / 200 of 467 steps | killed | 0.40 @ n=50; 0.42 @ n=150 | supported | adopt |
| exp-10 | 1714 | 5.12 | sft | exp-09 | gsm8k-train 10-shot (7463) | 1e-5 / 80 steps | completed | 0.46 @ n=50; 0.4933 @ n=150 | supported | adopt |
| exp-11 | 1861 | 5.40 | sft | exp-10 | gsm8k-train 10-shot (7463) | 5e-6 / 80 steps | completed | 0.44 @ n=50 | contradicted | reject |
| exp-12 | 2072 | 5.67 | sft | exp-10 | gsm8k-train 10-shot (7463) | 5e-6 / 40 steps | completed | 0.42 @ n=50 | contradicted | reject |

## Notes

- **Submitted model**: exp-10 (`runs/stage7_from_stage6ckpt200_lowlr80/checkpoint-80`,
  merged to `runs/merged_stage7_lowlr80`, copied to `final_model` at [1852]),
  verified at 0.4933 / n=150. exp-09's checkpoint held `final_model` between
  [1688] and [1852] and was verified at 0.420 / n=150, but that value was
  overwritten in `final_model_eval150.json`; the snapshot copy holds 0.4933.
- Every training card is one QLoRA (4-bit, `--use-4bit`) SFT stage of a single
  chained adapter; `setup.parent_checkpoint.origin` follows that chain. LoRA
  rank/alpha/dropout, seed, warmup_ratio and weight_decay come from argparse
  defaults in `scripts/train_gsm8k_sft.py`, which is not in the workspace
  snapshot, so they are null on every card.
- `outcome.official_accuracy` is never written; the schema key is absent from
  these cards by design.
- Six of the twelve gates were decided on 50 samples (stderr ~0.07) and two on 20
  samples (stderr ~0.10); several adopted deltas are inside one stderr. exp-07 was
  adopted on a 20-sample screen compared against a 50-sample incumbent number, a
  comparison the protocol does not license.
- The digest ends at [2149] with ~4.1 h of budget unspent.
